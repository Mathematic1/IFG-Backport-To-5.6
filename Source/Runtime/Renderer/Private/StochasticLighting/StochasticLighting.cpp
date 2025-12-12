// Copyright Epic Games, Inc. All Rights Reserved.

#include "StochasticLighting.h"
#include "RendererPrivate.h"
#include "Lumen/LumenScreenProbeGather.h"
#include "Lumen/LumenShortRangeAO.h"
#include "Lumen/LumenReflections.h"
#include "MegaLights/MegaLights.h"
#include "MegaLights/MegaLightsViewState.h"
#include "BasePassRendering.h"

static TAutoConsoleVariable<int32> CVarStochasticLightingFixedStateFrameIndex(
	TEXT("r.StochasticLighting.FixedStateFrameIndex"),
	-1,
	TEXT("Whether to override View.StateFrameIndex for debugging."),
	ECVF_RenderThreadSafe);

namespace StochasticLighting
{
	constexpr int32 TileSize = 8;
	constexpr int32 DownsampleFactor = 2;

	int32 GetStateFrameIndex(const FSceneViewState* ViewState)
	{
		int32 StateFrameIndex = CVarStochasticLightingFixedStateFrameIndex.GetValueOnRenderThread();
		if (StateFrameIndex < 0)
		{
			StateFrameIndex = ViewState ? ViewState->GetFrameIndex() : 0;
		}
		return StateFrameIndex;
	}

	bool IsStateFrameIndexOverridden()
	{
		return CVarStochasticLightingFixedStateFrameIndex.GetValueOnRenderThread() >= 0;
	}
}

class FStochasticLightingTileClassificationMarkCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FStochasticLightingTileClassificationMarkCS)
	SHADER_USE_PARAMETER_STRUCT(FStochasticLightingTileClassificationMarkCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureParameters, SceneTextures)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
		SHADER_PARAMETER_STRUCT_INCLUDE(FLumenFrontLayerTranslucencyGBufferParameters, FrontLayerTranslucencyGBufferParameters)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSubstrateGlobalUniformParameters, Substrate)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D, DepthHistoryTexture)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float4>, NormalAndShadingInfoHistory)
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<half>, MegaLightsNumFramesAccumulatedHistory)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<uint>, RWLumenTileBitmask)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<uint>, RWMegaLightsTileBitmask)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWDepthTexture)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float3>, RWNormalTexture)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, RWDownsampledSceneDepth)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<UNORM float3>, RWDownsampledWorldNormal)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWEncodedReprojectionVector)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<uint>, RWLumenPackedPixelData)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<uint>, RWMegaLightsPackedPixelData)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenScreenProbeGather::FTileClassifyParameters, ScreenProbeGatherTileClassifyParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenReflections::FCompositeParameters, ReflectionsCompositeParameters)
		SHADER_PARAMETER(uint32, ReflectionPass)
		SHADER_PARAMETER(FVector4f, HistoryScreenPositionScaleBias)
		SHADER_PARAMETER(FVector4f, HistoryUVMinMax)
		SHADER_PARAMETER(FVector4f, HistoryGatherUVMinMax)
		SHADER_PARAMETER(FVector4f, HistoryBufferSizeAndInvSize)
		SHADER_PARAMETER(FIntPoint, DownsampledViewMin2x2)
		SHADER_PARAMETER(FIntPoint, DownsampledViewSize2x2)
		SHADER_PARAMETER(uint32, LumenStochasticSampleMode)
		SHADER_PARAMETER(uint32, MegaLightsStochasticSampleMode)
		SHADER_PARAMETER(uint32, StochasticLightingStateFrameIndex)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FForwardLightUniformParameters, ForwardLightStruct)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FHairStrandsViewUniformParameters, HairStrands)
		SHADER_PARAMETER_STRUCT_REF(FBlueNoise, BlueNoise)
		SHADER_PARAMETER(uint32, TileEncoding)
		SHADER_PARAMETER(uint32, bRectPrimitive)
		SHADER_PARAMETER_ARRAY(FUintVector4, TileListBufferOffsets, [SUBSTRATE_TILE_TYPE_COUNT])
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer, TileDrawIndirectDataBufferUAV)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer, TileListBufferUAV)
		RDG_BUFFER_ACCESS(TileIndirectBuffer, ERHIAccess::IndirectArgs)
	END_SHADER_PARAMETER_STRUCT()

	class FCopyDepthAndNormal : SHADER_PERMUTATION_BOOL("COPY_DEPTH_AND_NORMAL");
	class FStochasticSampleOffset : SHADER_PERMUTATION_ENUM_CLASS("STOCHASTIC_SAMPLE_OFFSET", StochasticLighting::EStochasticSampleOffset);
	class FTileClassifyLumen : SHADER_PERMUTATION_BOOL("TILE_CLASSIFY_LUMEN");
	class FTileClassifyMegaLights : SHADER_PERMUTATION_BOOL("TILE_CLASSIFY_MEGALIGHTS");
	class FTileClassifySubstrate : SHADER_PERMUTATION_BOOL("TILE_CLASSIFY_SUBSTRATE");
	class FReprojectLumen : SHADER_PERMUTATION_BOOL("REPROJECT_LUMEN");
	class FReprojectMegaLights : SHADER_PERMUTATION_BOOL("REPROJECT_MEGALIGHTS");
	class FHistoryRejectBasedOnNormal : SHADER_PERMUTATION_BOOL("HISTORY_REJECT_BASED_ON_NORMAL");
	class FMaterialSource : SHADER_PERMUTATION_ENUM_CLASS("MATERIAL_SOURCE", StochasticLighting::EMaterialSource);
	class FOverflowTile : SHADER_PERMUTATION_BOOL("PERMUTATION_OVERFLOW_TILE");
	using FPermutationDomain = TShaderPermutationDomain<
		FCopyDepthAndNormal,
		FStochasticSampleOffset,
		FTileClassifyLumen,
		FTileClassifyMegaLights,
		FTileClassifySubstrate,
		FReprojectLumen,
		FReprojectMegaLights,
		FHistoryRejectBasedOnNormal,
		FMaterialSource,
		FOverflowTile>;

	static FPermutationDomain RemapPermutation(FPermutationDomain PermutationVector, EShaderPlatform InPlatform)
	{
		if (!Substrate::IsSubstrateEnabled())
		{
			PermutationVector.Set<FOverflowTile>(false);
			PermutationVector.Set<FTileClassifySubstrate>(false);
		}
		else if (!Substrate::IsSubstrateBlendableGBufferEnabled(InPlatform))
		{
			// Only available with Format=0 (blendable GBuffer)
			PermutationVector.Set<FTileClassifySubstrate>(false);
		}

		if (PermutationVector.Get<FStochasticSampleOffset>() == StochasticLighting::EStochasticSampleOffset::Both)
		{
			PermutationVector.Set<FMaterialSource>(StochasticLighting::EMaterialSource::GBuffer);
		}

		if (PermutationVector.Get<FMaterialSource>() != StochasticLighting::EMaterialSource::GBuffer)
		{
			PermutationVector.Set<FOverflowTile>(false);

			if (PermutationVector.Get<FMaterialSource>() == StochasticLighting::EMaterialSource::HairStrands)
			{
				PermutationVector.Set<FTileClassifyLumen>(false);
			}
			else
			{
				PermutationVector.Set<FCopyDepthAndNormal>(false);
				PermutationVector.Set<FStochasticSampleOffset>(StochasticLighting::EStochasticSampleOffset::None);
				PermutationVector.Set<FTileClassifyMegaLights>(false);
				PermutationVector.Set<FReprojectLumen>(false);
			}
		}

		if (PermutationVector.Get<FOverflowTile>())
		{
			PermutationVector.Set<FCopyDepthAndNormal>(false);
			PermutationVector.Set<FStochasticSampleOffset>(StochasticLighting::EStochasticSampleOffset::None);
			PermutationVector.Set<FTileClassifyMegaLights>(false);
		}

		if (!PermutationVector.Get<FTileClassifyLumen>())
		{
			PermutationVector.Set<FReprojectLumen>(false);
		}

		if (!PermutationVector.Get<FTileClassifyMegaLights>())
		{
			PermutationVector.Set<FReprojectMegaLights>(false);
		}

		if (!PermutationVector.Get<FReprojectLumen>())
		{
			PermutationVector.Set<FHistoryRejectBasedOnNormal>(false);
		}

		return PermutationVector;
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		FPermutationDomain PermutationVector(Parameters.PermutationId);

		if (RemapPermutation(PermutationVector, Parameters.Platform) != PermutationVector)
		{
			return false;
		}

		return DoesPlatformSupportLumenGI(Parameters.Platform) || MegaLights::ShouldCompileShaders(Parameters.Platform);
	}

	static int32 GetGroupSize()
	{
		return 8;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		FForwardLightingParameters::ModifyCompilationEnvironment(Parameters.Platform, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());

		FPermutationDomain PermutationVector(Parameters.PermutationId);
		if (PermutationVector.Get<FMaterialSource>() == StochasticLighting::EMaterialSource::FrontLayerGBuffer)
		{
			OutEnvironment.SetDefine(TEXT("FRONT_LAYER_TRANSLUCENCY"), 1);
		}
	}
};

IMPLEMENT_GLOBAL_SHADER(FStochasticLightingTileClassificationMarkCS, "/Engine/Private/StochasticLighting/StochasticLightingTileClassification.usf", "StochasticLightingTileClassificationMarkCS", SF_Compute);

StochasticLighting::FContext::FContext(
	FRDGBuilder& InGraphBuilder,
	const FMinimalSceneTextures& InSceneTextures,
	const FLumenFrontLayerTranslucencyGBufferParameters& InFrontLayerTranslucencyGBuffer,
	StochasticLighting::EMaterialSource InMaterialSource)
	: GraphBuilder(InGraphBuilder)
	, SceneTextures(InSceneTextures)
	, FrontLayerTranslucencyGBuffer(InFrontLayerTranslucencyGBuffer)
	, MaterialSource(InMaterialSource)
{}

void StochasticLighting::FContext::Validate(const StochasticLighting::FRunConfig& RunConfig) const
{
	if (RunConfig.bSubstrateOverflow)
	{
		check(MaterialSource == EMaterialSource::GBuffer);
	}

	if (RunConfig.bCopyDepthAndNormal)
	{
		check(DepthHistoryUAV && NormalHistoryUAV);
	}

	if (RunConfig.bDownsampleDepthAndNormal)
	{
		check(DownsampledSceneDepthUAV && DownsampledWorldNormalUAV);
	}

	if (RunConfig.bTileClassifyLumen)
	{
		check(LumenTileBitmaskUAV);
	}

	if (RunConfig.bTileClassifyMegaLights)
	{
		check(MegaLightsTileBitmaskUAV);
	}

	if (RunConfig.bReprojectLumen)
	{
		check(RunConfig.bTileClassifyLumen && EncodedReprojectionVectorUAV && LumenPackedPixelDataUAV);
	}

	if (RunConfig.bReprojectMegaLights)
	{
		check(RunConfig.bTileClassifyMegaLights && EncodedReprojectionVectorUAV && MegaLightsPackedPixelDataUAV);
	}
}

void StochasticLighting::FContext::Run(const FViewInfo& View, EReflectionsMethod ViewReflectionsMethod, const FRunConfig& RunConfig)
{
	Validate(RunConfig);

	FBlueNoise BlueNoise = GetBlueNoiseGlobalParameters();
	TUniformBufferRef<FBlueNoise> BlueNoiseUniformBuffer = CreateUniformBufferImmediate(BlueNoise, EUniformBufferUsage::UniformBuffer_SingleDraw);

	FVector4f HistoryScreenPositionScaleBias = FVector4f(1.0f, 1.0f, 0.0f, 0.0f);
	FVector4f HistoryUVMinMax = FVector4f::Zero();
	FVector4f HistoryGatherUVMinMax = FVector4f::Zero();
	FVector4f HistoryBufferSizeAndInvSize = FVector4f::Zero();
	FRDGTextureRef DepthHistoryTexture = nullptr;
	FRDGTextureRef NormalAndShadingInfoHistory = nullptr;
	FRDGTextureRef MegaLightsNumFramesAccumulatedHistory = nullptr;
	
	if (View.ViewState)
	{
		const bool bIsHairStrands = MaterialSource == EMaterialSource::HairStrands;
		const FStochasticLightingViewState& StochasticLightingViewState = View.ViewState->StochasticLighting;

		if (!View.bCameraCut && !View.bPrevTransformsReset)
		{
			HistoryScreenPositionScaleBias = StochasticLightingViewState.HistoryScreenPositionScaleBias;
			HistoryUVMinMax = StochasticLightingViewState.HistoryUVMinMax;
			HistoryGatherUVMinMax = StochasticLightingViewState.HistoryGatherUVMinMax;
			HistoryBufferSizeAndInvSize = StochasticLightingViewState.HistoryBufferSizeAndInvSize;

			if (bIsHairStrands)
			{
			}
			else
			{
				if (StochasticLightingViewState.SceneDepthHistory)
				{
					DepthHistoryTexture = GraphBuilder.RegisterExternalTexture(StochasticLightingViewState.SceneDepthHistory);
				}

				if (StochasticLightingViewState.SceneNormalHistory)
				{
					NormalAndShadingInfoHistory = GraphBuilder.RegisterExternalTexture(StochasticLightingViewState.SceneNormalHistory);
				}
			}
		}
	}

	const bool bTileClassifySubstrate = RunConfig.bTileClassifySubstrate && MaterialSource == EMaterialSource::GBuffer;
	const bool bHistoryRejectBasedOnNormal = RunConfig.bReprojectLumen && LumenScreenProbeGather::UseRejectBasedOnNormal() && NormalAndShadingInfoHistory;

	const FIntPoint DownsampledBufferSize2x1 = FIntPoint::DivideAndRoundUp(SceneTextures.Config.Extent, FIntPoint(2, 1));
	const FIntPoint DownsampledViewMin2x1 = FIntPoint::DivideAndRoundUp(View.ViewRect.Min, FIntPoint(2, 1));
	const FIntPoint DownsampledViewSize2x1 = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), FIntPoint(2, 1));
	const FIntPoint DownsampledBufferSize2x2 = FIntPoint::DivideAndRoundUp(SceneTextures.Config.Extent, 2);
	const FIntPoint DownsampledViewMin2x2 = FIntPoint::DivideAndRoundUp(View.ViewRect.Min, 2);
	const FIntPoint DownsampledViewSize2x2 = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), 2);
	
	const uint32 LumenStochasticSampleMode = uint32(LumenScreenProbeGather::IsUsingDownsampledDepthAndNormal(View) ? EStochasticSampleOffset::DownsampleFactor2x2 : EStochasticSampleOffset::None);
	
	uint32 MegaLightsStochasticSampleMode = (uint32)EStochasticSampleOffset::None;
	if (RunConfig.bTileClassifyMegaLights)
	{
		const FIntPoint MegaLightsDownsampleFactor = FIntPoint(1, 1);
		if (MegaLightsDownsampleFactor.X == 2)
		{
			if (MegaLightsDownsampleFactor.Y == 2)
			{
				MegaLightsStochasticSampleMode = (uint32)EStochasticSampleOffset::DownsampleFactor2x2;
			}
			else
			{
				MegaLightsStochasticSampleMode = (uint32)EStochasticSampleOffset::DownsampleFactor2x1;
			}
		}
		else
		{
			MegaLightsStochasticSampleMode = (uint32)EStochasticSampleOffset::None;
		}
	}

	int32 StateFrameIndex = GetStateFrameIndex(View.ViewState);
	if (RunConfig.StateFrameIndexOverride >= 0)
	{
		StateFrameIndex = RunConfig.StateFrameIndexOverride;
	}

	EStochasticSampleOffset StochasticSampleOffset = EStochasticSampleOffset::None;
	if (RunConfig.bDownsampleDepthAndNormal)
	{
		StochasticSampleOffset = EStochasticSampleOffset::DownsampleFactor2x2;
	}

	if ((RunConfig.bReprojectLumen || RunConfig.bReprojectMegaLights) && !DepthHistoryTexture)
	{
		DepthHistoryTexture = GSystemTextures.GetDepthDummy(GraphBuilder);
		NormalAndShadingInfoHistory = GSystemTextures.GetBlackDummy(GraphBuilder);
	}

	if (RunConfig.bReprojectMegaLights && !MegaLightsNumFramesAccumulatedHistory)
	{
		MegaLightsNumFramesAccumulatedHistory = GSystemTextures.GetBlackDummy(GraphBuilder);
	}

	FStochasticLightingTileClassificationMarkCS::FPermutationDomain PermutationVector;
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FCopyDepthAndNormal>(RunConfig.bCopyDepthAndNormal);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FStochasticSampleOffset>(StochasticSampleOffset);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FTileClassifyLumen>(RunConfig.bTileClassifyLumen);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FTileClassifyMegaLights>(RunConfig.bTileClassifyMegaLights);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FTileClassifySubstrate>(bTileClassifySubstrate);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FReprojectLumen>(RunConfig.bReprojectLumen);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FReprojectMegaLights>(RunConfig.bReprojectMegaLights);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FHistoryRejectBasedOnNormal>(bHistoryRejectBasedOnNormal);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FMaterialSource>(MaterialSource);
	PermutationVector.Set<FStochasticLightingTileClassificationMarkCS::FOverflowTile>(RunConfig.bSubstrateOverflow);
	PermutationVector = FStochasticLightingTileClassificationMarkCS::RemapPermutation(PermutationVector, View.GetShaderPlatform());
	auto ComputeShader = View.ShaderMap->GetShader<FStochasticLightingTileClassificationMarkCS>(PermutationVector);

	FStochasticLightingTileClassificationMarkCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FStochasticLightingTileClassificationMarkCS::FParameters>();
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->SceneTextures = GetSceneTextureParameters(GraphBuilder, SceneTextures.UniformBuffer);
	PassParameters->SceneTexturesStruct = SceneTextures.UniformBuffer;
	PassParameters->FrontLayerTranslucencyGBufferParameters = FrontLayerTranslucencyGBuffer;
	PassParameters->Substrate = Substrate::BindSubstrateGlobalUniformParameters(View);
	PassParameters->DepthHistoryTexture = DepthHistoryTexture;
	PassParameters->NormalAndShadingInfoHistory = NormalAndShadingInfoHistory;
	PassParameters->MegaLightsNumFramesAccumulatedHistory = MegaLightsNumFramesAccumulatedHistory;
	PassParameters->RWDepthTexture = DepthHistoryUAV;
	PassParameters->RWNormalTexture = NormalHistoryUAV;
	PassParameters->RWDownsampledSceneDepth = DownsampledSceneDepthUAV;
	PassParameters->RWDownsampledWorldNormal = DownsampledWorldNormalUAV;
	PassParameters->RWLumenTileBitmask = LumenTileBitmaskUAV;
	PassParameters->RWMegaLightsTileBitmask = MegaLightsTileBitmaskUAV;
	PassParameters->RWEncodedReprojectionVector = EncodedReprojectionVectorUAV;
	PassParameters->RWLumenPackedPixelData = LumenPackedPixelDataUAV;
	PassParameters->RWMegaLightsPackedPixelData = MegaLightsPackedPixelDataUAV;
	LumenScreenProbeGather::SetupTileClassifyParameters(View, PassParameters->ScreenProbeGatherTileClassifyParameters);
	LumenReflections::SetupCompositeParameters(View, PassParameters->ReflectionsCompositeParameters);
	PassParameters->ReflectionPass = (uint32)(MaterialSource == StochasticLighting::EMaterialSource::FrontLayerGBuffer ? ELumenReflectionPass::FrontLayerTranslucency : ELumenReflectionPass::Opaque);
	PassParameters->HistoryScreenPositionScaleBias = HistoryScreenPositionScaleBias;
	PassParameters->HistoryUVMinMax = HistoryUVMinMax;
	PassParameters->HistoryGatherUVMinMax = HistoryGatherUVMinMax;
	PassParameters->HistoryBufferSizeAndInvSize = HistoryBufferSizeAndInvSize;
	PassParameters->DownsampledViewMin2x2 = DownsampledViewMin2x2;
	PassParameters->DownsampledViewSize2x2 = DownsampledViewSize2x2;
	PassParameters->LumenStochasticSampleMode = LumenStochasticSampleMode;
	PassParameters->MegaLightsStochasticSampleMode = MegaLightsStochasticSampleMode;
	PassParameters->StochasticLightingStateFrameIndex = StateFrameIndex;
	PassParameters->ForwardLightStruct = View.ForwardLightingResources.ForwardLightUniformBuffer;
	PassParameters->HairStrands = HairStrands::BindHairStrandsViewUniformParameters(View);
	PassParameters->BlueNoise = BlueNoiseUniformBuffer;

	if (bTileClassifySubstrate)
	{
		const FSubstrateViewData* SubstrateViewData = &View.SubstrateViewData;
		PassParameters->TileDrawIndirectDataBufferUAV = SubstrateViewData->ClassificationTileDrawIndirectBufferUAV;
		PassParameters->TileListBufferUAV = SubstrateViewData->ClassificationTileListBufferUAV;
		PassParameters->TileEncoding = SubstrateViewData->TileEncoding;
		PassParameters->bRectPrimitive = GRHISupportsRectTopology ? 1 : 0;
		for (uint32 TileType = 0; TileType < SUBSTRATE_TILE_TYPE_COUNT; ++TileType)
		{
			PassParameters->TileListBufferOffsets[TileType] = FUintVector4(SubstrateViewData->ClassificationTileListBufferOffset[TileType], 0, 0, 0);
		}
	}

	if (RunConfig.bSubstrateOverflow)
	{
		PassParameters->TileIndirectBuffer = View.SubstrateViewData.ClosureTileDispatchIndirectBuffer;
		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("TileClassificationMark(Overflow)"),
			RunConfig.ComputePassFlags,
			ComputeShader,
			PassParameters,
			View.SubstrateViewData.ClosureTileDispatchIndirectBuffer,
			Substrate::GetClosureTileIndirectArgsOffset(/*InDownsampleFactor*/ 1));
	}
	else
	{
		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("TileClassificationMark"),
			RunConfig.ComputePassFlags,
			ComputeShader,
			PassParameters,
			FComputeShaderUtils::GetGroupCount(View.ViewRect.Size(), FStochasticLightingTileClassificationMarkCS::GetGroupSize()));
	}

	if (bTileClassifySubstrate)
	{
		// Sanity check
		check(!RunConfig.bSubstrateOverflow);
	}
}

static bool InternalRequiresStochasticLightingPass(const FViewFamilyInfo& ViewFamily, EDiffuseIndirectMethod InDiffuseIndirectMethod, EReflectionsMethod InReflectionsMethod)
{
	return InDiffuseIndirectMethod == EDiffuseIndirectMethod::Lumen
		|| InReflectionsMethod == EReflectionsMethod::Lumen;
}

bool FDeferredShadingSceneRenderer::RequiresStochasticLightingPass()
{
	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		const FPerViewPipelineState& ViewPipelineState = GetViewPipelineState(Views[ViewIndex]);
		if (InternalRequiresStochasticLightingPass(ViewFamily, ViewPipelineState.DiffuseIndirectMethod, ViewPipelineState.ReflectionsMethod))
		{
			return true;
		}
	}

	return false;
}

/**
 * Load GBuffer data once and transform it for subsequent lighting passes
 * This includes full res depth and normal copy for opaque before it gets overwritten by water or other translucency writing depth
 */
void FDeferredShadingSceneRenderer::StochasticLightingTileClassificationMark(FRDGBuilder& GraphBuilder, FLumenSceneFrameTemporaries& FrameTemporaries, const FSceneTextures& SceneTextures)
{
	const ERDGPassFlags ComputePassFlags = ERDGPassFlags::Compute;

	bool bNeedsClear = true;

	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];
		const FPerViewPipelineState& ViewPipelineState = GetViewPipelineState(View);

		if (InternalRequiresStochasticLightingPass(ViewFamily, ViewPipelineState.DiffuseIndirectMethod, ViewPipelineState.ReflectionsMethod))
		{
			const FSceneTextureParameters& SceneTextureParameters = GetSceneTextureParameters(GraphBuilder, SceneTextures);
			const uint32 ClosureCount = Substrate::GetSubstrateMaxClosureCount(View);
			const bool bCopyDepthAndNormal = View.ViewState && !View.bStatePrevViewInfoIsReadOnly;
			const ELumenFinalGatherMethod LumenFinalGatherMethod = Lumen::GetFinalGatherMethod(ViewFamily, ShaderPlatform);
			const bool bLumenScreenProbeGather = ViewPipelineState.DiffuseIndirectMethod == EDiffuseIndirectMethod::Lumen && LumenFinalGatherMethod == ELumenFinalGatherMethod::ScreenProbeGather;
			const bool bLumenIrradianceFieldGather = ViewPipelineState.DiffuseIndirectMethod == EDiffuseIndirectMethod::Lumen && LumenFinalGatherMethod == ELumenFinalGatherMethod::IrradianceFieldGather;
			const bool bTileClassifyLumen = bLumenScreenProbeGather || ViewPipelineState.ReflectionsMethod == EReflectionsMethod::Lumen;
			const bool bTileClassifyMegaLights = MegaLights::IsEnabled(ViewFamily);
			const bool bNeedsReprojection = bLumenScreenProbeGather || bTileClassifyMegaLights;
			const bool bDownsampleDepthAndNormal = (bLumenScreenProbeGather && LumenScreenProbeGather::IsUsingDownsampledDepthAndNormal(View))
				|| (bLumenIrradianceFieldGather && LumenIrradianceFieldGather::IsUsingDownsampledDepthAndNormal(View));

			FRDGTextureRef DepthHistory = nullptr;
			FRDGTextureRef NormalHistory = nullptr;
			if (bCopyDepthAndNormal)
			{
				DepthHistory = FrameTemporaries.DepthHistory.CreateSharedRT(GraphBuilder,
					FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_R32_FLOAT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
					FrameTemporaries.ViewExtent,
					TEXT("StochasticLighting.DepthHistory"));

				NormalHistory = FrameTemporaries.NormalHistory.CreateSharedRT(GraphBuilder,
					FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_A2B10G10R10, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
					FrameTemporaries.ViewExtent,
					TEXT("StochasticLighting.NormalAndShadingInfoHistory"));
			}

			FRDGTextureRef DownsampledSceneDepth = nullptr;
			FRDGTextureRef DownsampledWorldNormal = nullptr;
			if (bDownsampleDepthAndNormal)
			{
				FIntPoint DownsampledBufferSize = FIntPoint::DivideAndRoundUp(SceneTextures.Config.Extent, FIntPoint(2, 2));

				DownsampledSceneDepth = FrameTemporaries.DownsampledSceneDepth.CreateSharedRT(GraphBuilder,
					FRDGTextureDesc::Create2D(DownsampledBufferSize, PF_R32_FLOAT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
					FrameTemporaries.ViewExtent,
					TEXT("StochasticLighting.DownsampledSceneDepth"));

				DownsampledWorldNormal = FrameTemporaries.DownsampledWorldNormal.CreateSharedRT(GraphBuilder,
					FRDGTextureDesc::Create2D(DownsampledBufferSize, PF_A2B10G10R10, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
					FrameTemporaries.ViewExtent,
					TEXT("StochasticLighting.DownsampledWorldNormal"));
			}

			FRDGTextureRef LumenTileBitmask = nullptr;
			if (bTileClassifyLumen)
			{
				const FIntPoint BufferSize = Substrate::GetSubstrateTextureResolution(View, SceneTextures.Config.Extent);
				const FIntPoint BufferSizeInTiles = FIntPoint::DivideAndRoundUp(BufferSize, StochasticLighting::TileSize);

				LumenTileBitmask = FrameTemporaries.LumenTileBitmask.CreateSharedRT(GraphBuilder,
					FRDGTextureDesc::Create2DArray(BufferSizeInTiles, PF_R8_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV, ClosureCount),
					FrameTemporaries.ViewExtent,
					TEXT("StochasticLighting.LumenTileBitmask"));
			}

			FRDGTextureRef MegaLightsTileBitmask = nullptr;
			if (bTileClassifyMegaLights)
			{
				const FIntPoint BufferSizeInTiles = FIntPoint::DivideAndRoundUp(SceneTextures.Config.Extent, StochasticLighting::TileSize);

				MegaLightsTileBitmask = FrameTemporaries.MegaLightsTileBitmask.CreateSharedRT(GraphBuilder,
					FRDGTextureDesc::Create2D(BufferSizeInTiles, PF_R8_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
					FrameTemporaries.ViewExtent,
					TEXT("StochasticLighting.MegaLightsTileBitmask"));
			}

			FRDGTextureRef EncodedReprojectionVector = nullptr;
			FRDGTextureRef LumenPackedPixelData = nullptr;
			FRDGTextureRef MegaLightsPackedPixelData = nullptr;
			if (bNeedsReprojection)
			{
				EncodedReprojectionVector = FrameTemporaries.EncodedReprojectionVector.CreateSharedRT(GraphBuilder,
					FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_R32_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
					FrameTemporaries.ViewExtent,
					TEXT("StochasticLighting.EncodedReprojectionVector"));

				if (bLumenScreenProbeGather)
				{
					LumenPackedPixelData = FrameTemporaries.LumenPackedPixelData.CreateSharedRT(GraphBuilder,
						FRDGTextureDesc::Create2DArray(SceneTextures.Config.Extent, PF_R8_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV, ClosureCount),
						FrameTemporaries.ViewExtent,
						TEXT("StochasticLighting.LumenPackedPixelData"));
				}

				if (bTileClassifyMegaLights)
				{
					MegaLightsPackedPixelData = FrameTemporaries.MegaLightsPackedPixelData.CreateSharedRT(GraphBuilder,
						FRDGTextureDesc::Create2D(SceneTextures.Config.Extent, PF_R8_UINT, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV),
						FrameTemporaries.ViewExtent,
						TEXT("StochasticLighting.MegaLightsPackedPixelData"));
				}
			}

			FLumenFrontLayerTranslucencyGBufferParameters FrontLayerTranslucencyGBuffer;
			FrontLayerTranslucencyGBuffer.FrontLayerTranslucencyNormal = nullptr;
			FrontLayerTranslucencyGBuffer.FrontLayerTranslucencySceneDepth = nullptr;

			FRDGTextureUAVRef DepthHistoryUAV = DepthHistory ? GraphBuilder.CreateUAV(DepthHistory, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef NormalHistoryUAV = NormalHistory ? GraphBuilder.CreateUAV(NormalHistory, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef DownsampledSceneDepthUAV = DownsampledSceneDepth ? GraphBuilder.CreateUAV(DownsampledSceneDepth, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef DownsampledWorldNormalUAV = DownsampledWorldNormal ? GraphBuilder.CreateUAV(DownsampledWorldNormal, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef LumenTileBitmaskUAV = LumenTileBitmask ? GraphBuilder.CreateUAV(LumenTileBitmask, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef MegaLightsTileBitmaskUAV = MegaLightsTileBitmask ? GraphBuilder.CreateUAV(MegaLightsTileBitmask, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef EncodedReprojectionVectorUAV = EncodedReprojectionVector ? GraphBuilder.CreateUAV(EncodedReprojectionVector, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef LumenPackedPixelDataUAV = LumenPackedPixelData ? GraphBuilder.CreateUAV(LumenPackedPixelData, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;
			FRDGTextureUAVRef MegaLightsPackedPixelDataUAV = MegaLightsPackedPixelData ? GraphBuilder.CreateUAV(MegaLightsPackedPixelData, ERDGUnorderedAccessViewFlags::SkipBarrier) : nullptr;

			StochasticLighting::FRunConfig RunConfig;
			RunConfig.ComputePassFlags = ComputePassFlags;
			RunConfig.bCopyDepthAndNormal = DepthHistoryUAV != nullptr;
			RunConfig.bDownsampleDepthAndNormal = DownsampledSceneDepthUAV != nullptr;
			RunConfig.bTileClassifyLumen = LumenTileBitmaskUAV != nullptr;
			RunConfig.bTileClassifyMegaLights = MegaLightsTileBitmaskUAV != nullptr;
			RunConfig.bReprojectLumen = LumenPackedPixelDataUAV != nullptr;
			RunConfig.bReprojectMegaLights = MegaLightsPackedPixelDataUAV != nullptr;

			// TODO: share context between views
			StochasticLighting::FContext StochasticLightingContext(GraphBuilder, SceneTextures, FrontLayerTranslucencyGBuffer, StochasticLighting::EMaterialSource::GBuffer);
			StochasticLightingContext.DepthHistoryUAV = DepthHistoryUAV;
			StochasticLightingContext.NormalHistoryUAV = NormalHistoryUAV;
			StochasticLightingContext.DownsampledSceneDepthUAV = DownsampledSceneDepthUAV;
			StochasticLightingContext.DownsampledWorldNormalUAV = DownsampledWorldNormalUAV;
			StochasticLightingContext.LumenTileBitmaskUAV = LumenTileBitmaskUAV;
			StochasticLightingContext.MegaLightsTileBitmaskUAV = MegaLightsTileBitmaskUAV;
			StochasticLightingContext.EncodedReprojectionVectorUAV = EncodedReprojectionVectorUAV;
			StochasticLightingContext.LumenPackedPixelDataUAV = LumenPackedPixelDataUAV;
			StochasticLightingContext.MegaLightsPackedPixelDataUAV = MegaLightsPackedPixelDataUAV;

			if (Lumen::SupportsMultipleClosureEvaluation(View))
			{
				if (LumenPackedPixelData && ClosureCount > 1 && bNeedsClear)
				{
					const uint32 LumenInvalidPackedPixelData = 0x30;

					// Initialize LumenPackedPixelData to invalid value for all pixels belonging to slice>0, i.e. closure with index > 0. This is necessary because:
					// 1) The classification is dispatched only on valid tiles. For closure>0, the LumenPackedPixelData won't be initialized otherwise
					// 2) The temporal reprojection pass uses LumenPackedPixelData to update history value (in particular NumFamesAccumulated), which are used next frame to prune invalid history data.
					// Without this, LumenScreenProbeGather will fetch invalid/uninitialized history data for closure>0, causing visual artifacts
					FRDGTextureUAVRef LumenPackedPixelDataOverflowUAV = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(LumenPackedPixelData, 0/*InMipLevel*/, LumenPackedPixelData->Desc.Format, 1/*InFirstArraySlice*/, ClosureCount - 1u/*InNumArraySlices*/));

					// This value needs to be kept in sync with StochasticLightingCommon.ush
					AddClearUAVPass(GraphBuilder, LumenPackedPixelDataOverflowUAV, LumenInvalidPackedPixelData, RunConfig.ComputePassFlags);
					bNeedsClear = false;
				}
			}
			
			StochasticLightingContext.Run(View, ViewPipelineState.ReflectionsMethod, RunConfig);

			if (Lumen::SupportsMultipleClosureEvaluation(View))
			{
				StochasticLighting::FRunConfig OverflowTileRunConfig;
				OverflowTileRunConfig.ComputePassFlags = ComputePassFlags;
				OverflowTileRunConfig.bSubstrateOverflow = true;
				OverflowTileRunConfig.bTileClassifyLumen = LumenTileBitmaskUAV != nullptr;
				OverflowTileRunConfig.bReprojectLumen = LumenPackedPixelDataUAV != nullptr;

				StochasticLightingContext.Run(View, ViewPipelineState.ReflectionsMethod, OverflowTileRunConfig);
			}
		}
	}
}

void FDeferredShadingSceneRenderer::QueueExtractStochasticLighting(FRDGBuilder& GraphBuilder, FLumenSceneFrameTemporaries& FrameTemporaries, const FMinimalSceneTextures& SceneTextures)
{
	for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
	{
		FViewInfo& View = Views[ViewIndex];

		if (View.ViewState && !View.bStatePrevViewInfoIsReadOnly)
		{
			FStochasticLightingViewState& ViewState = View.ViewState->StochasticLighting;

			if (FrameTemporaries.DepthHistory.GetRenderTarget())
			{
				GraphBuilder.QueueTextureExtraction(FrameTemporaries.DepthHistory.GetRenderTarget(), &ViewState.SceneDepthHistory);
			}
			else
			{
				ViewState.SceneDepthHistory = nullptr;
			}

			if (FrameTemporaries.NormalHistory.GetRenderTarget())
			{
				GraphBuilder.QueueTextureExtraction(FrameTemporaries.NormalHistory.GetRenderTarget(), &ViewState.SceneNormalHistory);
			}
			else
			{
				ViewState.SceneNormalHistory = nullptr;
			}

			ViewState.HistoryScreenPositionScaleBias = View.GetScreenPositionScaleBias(SceneTextures.Config.Extent, View.ViewRect);

			const FVector2f InvBufferSize(1.0f / SceneTextures.Config.Extent.X, 1.0f / SceneTextures.Config.Extent.Y);

			ViewState.HistoryUVMinMax = FVector4f(
				View.ViewRect.Min.X * InvBufferSize.X,
				View.ViewRect.Min.Y * InvBufferSize.Y,
				View.ViewRect.Max.X * InvBufferSize.X,
				View.ViewRect.Max.Y * InvBufferSize.Y);

			// Clamp gather4 to a valid bilinear footprint in order to avoid sampling outside of valid bounds
			ViewState.HistoryGatherUVMinMax = FVector4f(
				(View.ViewRect.Min.X + 0.51f) * InvBufferSize.X,
				(View.ViewRect.Min.Y + 0.51f) * InvBufferSize.Y,
				(View.ViewRect.Max.X - 0.51f) * InvBufferSize.X,
				(View.ViewRect.Max.Y - 0.51f) * InvBufferSize.Y);

			ViewState.HistoryBufferSizeAndInvSize = FVector4f(
				SceneTextures.Config.Extent.X,
				SceneTextures.Config.Extent.Y,
				1.0f / SceneTextures.Config.Extent.X,
				1.0f / SceneTextures.Config.Extent.Y);
		}
	}
}