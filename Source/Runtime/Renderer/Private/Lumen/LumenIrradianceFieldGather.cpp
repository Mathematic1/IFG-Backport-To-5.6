// Copyright Epic Games, Inc. All Rights Reserved.

#include "RendererPrivate.h"
#include "ScenePrivate.h"
#include "SceneUtils.h"
#include "PipelineStateCache.h"
#include "ShaderParameterStruct.h"
#include "PixelShaderUtils.h"
#include "ReflectionEnvironment.h"
#include "DistanceFieldAmbientOcclusion.h"
#include "ScreenSpaceDenoise.h"
#include "LumenRadianceCache.h"
#include "LumenTracingUtils.h"
#include "LumenReflections.h"
#include "LumenScreenProbeGather.h"
#include "LumenShortRangeAO.h"

int32 GLumenIrradianceFieldNumClipmaps = 4;
FAutoConsoleVariableRef CVarLumenIrradianceFieldNumClipmaps(
	TEXT("r.Lumen.IrradianceFieldGather.NumClipmaps"),
	GLumenIrradianceFieldNumClipmaps,
	TEXT("Number of radiance cache clipmaps."),
	ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldClipmapWorldExtent = 5000.0f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldClipmapWorldExtent(
	TEXT("r.Lumen.IrradianceFieldGather.ClipmapWorldExtent"),
	GLumenIrradianceFieldClipmapWorldExtent,
	TEXT("World space extent of the first clipmap"),
	ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldClipmapDistributionBase = 2.0f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldClipmapDistributionBase(
	TEXT("r.Lumen.IrradianceFieldGather.ClipmapDistributionBase"),
	GLumenIrradianceFieldClipmapDistributionBase,
	TEXT("Base of the Pow() that controls the size of each successive clipmap relative to the first."),
	ECVF_RenderThreadSafe
);

static TAutoConsoleVariable<int32> CVarLumenIrradianceFieldNumProbesToTraceBudget(
	TEXT("r.Lumen.IrradianceFieldGather.NumProbesToTraceBudget"),
	100,
	TEXT("Number of radiance cache probes that can be updated per frame."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldGridResolution = 64;
FAutoConsoleVariableRef CVarLumenIrradianceFieldResolution(
	TEXT("r.Lumen.IrradianceFieldGather.GridResolution"),
	GLumenIrradianceFieldGridResolution,
	TEXT("Resolution of the probe placement grid within each clipmap"),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeResolution = 16;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeResolution(
	TEXT("r.Lumen.IrradianceFieldGather.ProbeResolution"),
	GLumenIrradianceFieldProbeResolution,
	TEXT("Resolution of the probe's 2d radiance layout.  The number of rays traced for the probe will be ProbeResolution ^ 2"),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeIrradianceResolution = 6;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeIrradianceResolution(
	TEXT("r.Lumen.IrradianceFieldGather.IrradianceProbeResolution"),
	GLumenIrradianceFieldProbeIrradianceResolution,
	TEXT("Resolution of the probe's 2d irradiance layout."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeOcclusionResolution = 16;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeOcclusionResolution(
	TEXT("r.Lumen.IrradianceFieldGather.OcclusionProbeResolution"),
	GLumenIrradianceFieldProbeOcclusionResolution,
	TEXT("Resolution of the probe's 2d occlusion layout."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldNumMipmaps = 3;
FAutoConsoleVariableRef CVarLumenIrradianceFieldNumMipmaps(
	TEXT("r.Lumen.IrradianceFieldGather.NumMipmaps"),
	GLumenIrradianceFieldNumMipmaps,
	TEXT("Number of radiance cache mipmaps."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldProbeAtlasResolutionInProbes = 128;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeAtlasResolutionInProbes(
	TEXT("r.Lumen.IrradianceFieldGather.ProbeAtlasResolutionInProbes"),
	GLumenIrradianceFieldProbeAtlasResolutionInProbes,
	TEXT("Number of probes along one dimension of the probe atlas cache texture.  This controls the memory usage of the cache.  Overflow currently results in incorrect rendering."),
	ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldProbeOcclusionBias = .8f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldProbeOcclusionBias(
	TEXT("r.Lumen.IrradianceFieldGather.ProbeOcclusionBias"),
	GLumenIrradianceFieldProbeOcclusionBias,
	TEXT("Bias along the normal and toward the viewer to reduce self-occlusion artifacts from Probe Occlusion"),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldBlendProbeOutputs = 0;
FAutoConsoleVariableRef CVarLumenIrradianceFieldBlendProbeOutputs(
	TEXT("r.Lumen.IrradianceFieldGather.BlendProbeOutputs"),
	GLumenIrradianceFieldBlendProbeOutputs,
	TEXT("Whether to blend the outputs of irradiance probes to smooth out changes over time."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldShortRangeAORadius = 32.0f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldShortRangeAORadius(
	TEXT("r.Lumen.IrradianceFieldGather.ShortRangeAORadius"),
	GLumenIrradianceFieldShortRangeAORadius,
	TEXT("Radius of ShortRangeAO, in pixels"),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldStats = 0;
FAutoConsoleVariableRef CVarLumenIrradianceFieldStats(
	TEXT("r.Lumen.IrradianceFieldGather.RadianceCache.Stats"),
	GLumenIrradianceFieldStats,
	TEXT("GPU print out Radiance Cache update stats."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldInterpolateDownsampleFactor = 2;
FAutoConsoleVariableRef CVarLumenIrradianceFieldInterpolateDownsampleFactor(
	TEXT("r.Lumen.IrradianceFieldGather.InterpolateDownsampleFactor"),
	GLumenIrradianceFieldInterpolateDownsampleFactor,
	TEXT("Downsample factor for Irradiance Field interpolation and shading."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldAdaptivePlacement = 0;
FAutoConsoleVariableRef CVarLumenIrradianceFieldAdaptivePlacement(
	TEXT("r.Lumen.IrradianceFieldGather.AdaptivePlacement"),
	GLumenIrradianceFieldAdaptivePlacement,
	TEXT("Whether to place new probes where interpolation fails due to occlusion."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize = 8;
FAutoConsoleVariableRef CVarLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize(
	TEXT("r.Lumen.IrradianceFieldGather.AdaptivePlacement.InterpolationFeedbackTileSize"),
	GLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize,
	TEXT("One pixel in every screen tile will process interpolation failures for adaptive probe placement."),
	ECVF_RenderThreadSafe
);

float GLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular = .8f;
FAutoConsoleVariableRef CVarLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular(
	TEXT("r.Lumen.IrradianceFieldGather.MaxRoughnessToEvaluateRoughSpecular"),
	GLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular,
	TEXT("Maximum material roughness to evaluate rough specular for. Pixels with roughness larger than this will reuse diffuse GI as their specular GI."),
	ECVF_RenderThreadSafe
);

int32 GLumenIrradianceFieldMarkProbesDownsampleFactor = 2;
FAutoConsoleVariableRef CVarLumenIrradianceFieldMarkProbesDownsampleFactor(
	TEXT("r.Lumen.IrradianceFieldGather.MarkProbesDownsampleFactor"),
	GLumenIrradianceFieldMarkProbesDownsampleFactor,
	TEXT("Downsample factor (applied on top of InterpolateDownsampleFactor) used to mark the irradiance probes as used by the GBuffer. Downsampling too high can cause small geometry to flicker black."),
	ECVF_Scalability | ECVF_RenderThreadSafe
);

namespace LumenIrradianceFieldGather
{
	LumenRadianceCache::FRadianceCacheInputs SetupRadianceCacheInputs()
	{
		LumenRadianceCache::FRadianceCacheInputs Parameters = LumenRadianceCache::GetDefaultRadianceCacheInputs();
		Parameters.ReprojectionRadiusScale = 1.5f;
		Parameters.ClipmapWorldExtent = GLumenIrradianceFieldClipmapWorldExtent;
		Parameters.ClipmapDistributionBase = GLumenIrradianceFieldClipmapDistributionBase;
		Parameters.RadianceProbeClipmapResolution = FMath::Clamp(GLumenIrradianceFieldGridResolution, 1, 256);
		Parameters.ProbeAtlasResolutionInProbes = FIntPoint(GLumenIrradianceFieldProbeAtlasResolutionInProbes, GLumenIrradianceFieldProbeAtlasResolutionInProbes);
		Parameters.NumRadianceProbeClipmaps = FMath::Clamp(GLumenIrradianceFieldNumClipmaps, 1, LumenRadianceCache::MaxClipmaps);
		Parameters.RadianceProbeResolution = FMath::Max(GLumenIrradianceFieldProbeResolution, LumenRadianceCache::MinRadianceProbeResolution);
		Parameters.FinalProbeResolution = GLumenIrradianceFieldProbeResolution + 2 * (1 << (GLumenIrradianceFieldNumMipmaps - 1));
		Parameters.FinalRadianceAtlasMaxMip = GLumenIrradianceFieldNumMipmaps - 1;
		Parameters.CalculateIrradiance = 1;
		Parameters.BlendProbes = GLumenIrradianceFieldBlendProbeOutputs != 0 ? 1 : 0;
		Parameters.UseAdaptiveProbes = GLumenIrradianceFieldAdaptivePlacement != 0 ? 1 : 0;
		Parameters.IrradianceProbeResolution = GLumenIrradianceFieldProbeIrradianceResolution;
		Parameters.OcclusionProbeResolution = GLumenIrradianceFieldProbeOcclusionResolution;
		Parameters.NumProbesToTraceBudget = CVarLumenIrradianceFieldNumProbesToTraceBudget.GetValueOnRenderThread();
		Parameters.RadianceCacheStats = GLumenIrradianceFieldStats;
		// Avoid overflow with quick camera movements
		Parameters.NumFramesToKeepCachedProbes = 2;
		return Parameters;
	}

	bool IsUsingDownsampledDepthAndNormal(const FViewInfo& View)
	{
		return GLumenIrradianceFieldInterpolateDownsampleFactor != 1 || LumenShortRangeAO::GetDownsampleFactor() != 1;
	}
}

class FMarkProbesUsedByGBufferCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMarkProbesUsedByGBufferCS)
	SHADER_USE_PARAMETER_STRUCT(FMarkProbesUsedByGBufferCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenRadianceCache::FRadianceCacheMarkParameters, RadianceCacheMarkParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FLumenUpsampleParameters, UpsampleParameters)
		SHADER_PARAMETER(uint32, MarkProbesDownsampleFactor)
	END_SHADER_PARAMETER_STRUCT()

	class FInterpolateDownsampleFactor : SHADER_PERMUTATION_RANGE_INT("INTERPOLATE_DOWNSAMPLE_FACTOR", 1, 2);
	using FPermutationDomain = TShaderPermutationDomain<FInterpolateDownsampleFactor>;

public:

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static uint32 GetGroupSize()
	{
		return 8;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FMarkProbesUsedByGBufferCS, "/Engine/Private/Lumen/LumenIrradianceFieldGather.usf", "MarkProbesUsedByGBufferCS", SF_Compute);


class FInterpolateIrradianceProbesToPixelsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FInterpolateIrradianceProbesToPixelsCS)
	SHADER_USE_PARAMETER_STRUCT(FInterpolateIrradianceProbesToPixelsCS, FGlobalShader)

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<float3>, RWDiffuseIndirect)
		SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2DArray<float3>, RWRoughSpecularIndirect)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenRadianceCache::FRadianceCacheInterpolationParameters, RadianceCacheParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(LumenReflections::FCompositeParameters, ReflectionsCompositeParameters)
		SHADER_PARAMETER_STRUCT_INCLUDE(FLumenUpsampleParameters, UpsampleParameters)
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSceneTextureUniformParameters, SceneTexturesStruct)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FSubstrateGlobalUniformParameters, Substrate)
		SHADER_PARAMETER_RDG_UNIFORM_BUFFER(FReflectionUniformParameters, ReflectionStruct)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ProbeAdaptiveIndices)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, ProbeValid)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWProbeLastUsedFrame)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, RWProbeInterpolationMisses)
		SHADER_PARAMETER(uint32, InterpolationFeedbackTileWrapMask)
		SHADER_PARAMETER(FIntPoint, InterpolationFeedbackTileJitter)
		SHADER_PARAMETER(uint32, MaxNumProbes)
		SHADER_PARAMETER(float, ProbeOcclusionBias)
		SHADER_PARAMETER(float, MaxRoughnessToEvaluateRoughSpecular)
	END_SHADER_PARAMETER_STRUCT()

	class FInterpolateDownsampleFactor : SHADER_PERMUTATION_RANGE_INT("INTERPOLATE_DOWNSAMPLE_FACTOR", 1, 2);
	class FAdaptiveProbes : SHADER_PERMUTATION_BOOL("ADAPTIVE_PROBES");
	using FPermutationDomain = TShaderPermutationDomain<FInterpolateDownsampleFactor, FAdaptiveProbes>;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return DoesPlatformSupportLumenGI(Parameters.Platform);
	}

	static int32 GetGroupSize() 
	{
		return 8;
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), GetGroupSize());
	}
};

IMPLEMENT_GLOBAL_SHADER(FInterpolateIrradianceProbesToPixelsCS, "/Engine/Private/Lumen/LumenIrradianceFieldGather.usf", "InterpolateIrradianceProbesToPixelsCS", SF_Compute);

static void IrradianceFieldMarkUsedProbes(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FSceneTextures& SceneTextures,
	const LumenRadianceCache::FRadianceCacheMarkParameters& RadianceCacheMarkParameters,
	const FLumenUpsampleParameters& UpsampleParameters,
	ERDGPassFlags ComputePassFlags)
{
	FMarkProbesUsedByGBufferCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FMarkProbesUsedByGBufferCS::FParameters>();
	PassParameters->View = View.ViewUniformBuffer;
	PassParameters->SceneTexturesStruct = SceneTextures.UniformBuffer;
	PassParameters->RadianceCacheMarkParameters = RadianceCacheMarkParameters;
	PassParameters->UpsampleParameters = UpsampleParameters;
	PassParameters->MarkProbesDownsampleFactor = FMath::Clamp<uint32>(GLumenIrradianceFieldMarkProbesDownsampleFactor, 1, 8);

	FMarkProbesUsedByGBufferCS::FPermutationDomain PermutationVector;
	PermutationVector.Set<FMarkProbesUsedByGBufferCS::FInterpolateDownsampleFactor>(UpsampleParameters.DownsampleFactor);
	auto ComputeShader = View.ShaderMap->GetShader<FMarkProbesUsedByGBufferCS>(PermutationVector);

	FComputeShaderUtils::AddPass(
		GraphBuilder,
		RDG_EVENT_NAME("MarkProbesUsedByGBuffer InterpolateDownsample=%u, MarkDownsample=%u", UpsampleParameters.DownsampleFactor, PassParameters->MarkProbesDownsampleFactor),
		ComputePassFlags,
		ComputeShader,
		PassParameters,
		FComputeShaderUtils::GetGroupCount(FIntPoint::DivideAndRoundUp(UpsampleParameters.IntegrateViewSize, PassParameters->MarkProbesDownsampleFactor), FInterpolateIrradianceProbesToPixelsCS::GetGroupSize()));
}

FIntPoint GetFeedbackBufferTileJitter(uint32 TileSize, uint32 FrameIndex) 
{
	const uint32 TileSizeLog2 = FMath::CeilLogTwo(TileSize);
	const uint32 SequenceSize = FMath::Square(TileSize);
	const uint32 PixelIndex = FrameIndex % SequenceSize;
	const uint32 PixelAddress = ReverseBits(PixelIndex) >> (32U - 2 * TileSizeLog2);

	FIntPoint TileJitter;
	TileJitter.X = FMath::ReverseMortonCode2(PixelAddress);
	TileJitter.Y = FMath::ReverseMortonCode2(PixelAddress >> 1);

	return TileJitter;
}

DECLARE_GPU_STAT(LumenIrradianceFieldGather);

FSSDSignalTextures FDeferredShadingSceneRenderer::RenderLumenIrradianceFieldGather(
	FRDGBuilder& GraphBuilder,
	const FSceneTextures& SceneTextures,
	const FLumenSceneFrameTemporaries& FrameTemporaries,
	FRDGTextureRef LightingChannelsTexture,
	const FViewInfo& View,
	FLumenScreenSpaceBentNormalParameters& ScreenSpaceBentNormalParameters,
	FLumenUpsampleParameters& UpsampleParameters,
	LumenRadianceCache::FRadianceCacheInterpolationParameters& TranslucencyVolumeRadianceCacheParameters,
	ERDGPassFlags ComputePassFlags)
{
	RDG_EVENT_SCOPE_STAT(GraphBuilder, LumenIrradianceFieldGather, "LumenIrradianceFieldGather");
	RDG_GPU_STAT_SCOPE(GraphBuilder, LumenIrradianceFieldGather);

	const LumenRadianceCache::FRadianceCacheInputs RadianceCacheInputs = LumenIrradianceFieldGather::SetupRadianceCacheInputs();
	const int32 MaxNumProbes = RadianceCacheInputs.ProbeAtlasResolutionInProbes.X * RadianceCacheInputs.ProbeAtlasResolutionInProbes.Y;
	const int32 InterpolateDownsampleFactor = FMath::Clamp<int32>(GLumenIrradianceFieldInterpolateDownsampleFactor, 1, 2);
	const FIntPoint DownsampledBufferSize = FIntPoint::DivideAndRoundUp(SceneTextures.Config.Extent, InterpolateDownsampleFactor);

	{
		UpsampleParameters.DownsampleFactor = InterpolateDownsampleFactor;
		UpsampleParameters.IntegrateViewMin = FIntPoint::DivideAndRoundUp(View.ViewRect.Min, InterpolateDownsampleFactor);
		UpsampleParameters.IntegrateViewSize = FIntPoint::DivideAndRoundUp(View.ViewRect.Size(), InterpolateDownsampleFactor);
		UpsampleParameters.DownsampledBufferInvSize = FVector2f(1.0f) / DownsampledBufferSize;
		UpsampleParameters.ScreenProbeGatherStateFrameIndex = LumenScreenProbeGather::GetStateFrameIndex(View.ViewState);
	}

	FRDGTextureRef DownsampledSceneDepth = nullptr;
	FRDGTextureRef DownsampledSceneWorldNormal = nullptr;

	if (InterpolateDownsampleFactor != 1)
	{
		ensure(LumenIrradianceFieldGather::IsUsingDownsampledDepthAndNormal(View));
		DownsampledSceneDepth = FrameTemporaries.DownsampledSceneDepth.GetRenderTarget();
		DownsampledSceneWorldNormal = FrameTemporaries.DownsampledWorldNormal.GetRenderTarget();

		UpsampleParameters.DownsampledSceneDepth = DownsampledSceneDepth;
		UpsampleParameters.DownsampledSceneWorldNormal = DownsampledSceneWorldNormal;
	}

	FMarkUsedRadianceCacheProbes Callbacks;
	Callbacks.AddLambda([&SceneTextures, UpsampleParameters, ComputePassFlags](
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const LumenRadianceCache::FRadianceCacheMarkParameters& RadianceCacheMarkParameters)
		{
			IrradianceFieldMarkUsedProbes(
				GraphBuilder,
				View,
				SceneTextures,
				RadianceCacheMarkParameters,
				UpsampleParameters,
				ComputePassFlags);
		});

	LumenRadianceCache::FRadianceCacheInterpolationParameters RadianceCacheParameters;

	LumenRadianceCache::TInlineArray<LumenRadianceCache::FUpdateInputs> InputArray;
	LumenRadianceCache::TInlineArray<LumenRadianceCache::FUpdateOutputs> OutputArray;

	InputArray.Add(LumenRadianceCache::FUpdateInputs(
		RadianceCacheInputs,
		FRadianceCacheConfiguration(),
		View,
		nullptr,
		nullptr,
		FMarkUsedRadianceCacheProbes(),
		MoveTemp(Callbacks)));

	FRadianceCacheState& RadianceCacheState = View.ViewState->Lumen.RadianceCacheState;

	OutputArray.Add(LumenRadianceCache::FUpdateOutputs(
		RadianceCacheState,
		RadianceCacheParameters));

	LumenRadianceCache::FUpdateInputs TranslucencyVolumeRadianceCacheUpdateInputs = GetLumenTranslucencyGIVolumeRadianceCacheInputs(
		GraphBuilder,
		View, 
		FrameTemporaries,
		ComputePassFlags);

	if (TranslucencyVolumeRadianceCacheUpdateInputs.IsAnyCallbackBound())
	{
		InputArray.Add(TranslucencyVolumeRadianceCacheUpdateInputs);
		OutputArray.Add(LumenRadianceCache::FUpdateOutputs(
			View.ViewState->Lumen.TranslucencyVolumeRadianceCacheState,
			TranslucencyVolumeRadianceCacheParameters));
	}

	LumenRadianceCache::UpdateRadianceCaches(
		GraphBuilder, 
		FrameTemporaries,
		InputArray,
		OutputArray,
		Scene,
		ViewFamily,
		SceneTextures,
		LumenCardRenderer.bPropagateGlobalLightingChange,
		ComputePassFlags);

	if (LumenScreenProbeGather::UseShortRangeAmbientOcclusion(ViewFamily.EngineShowFlags))
	{
		FVector2f MaxScreenTraceFraction = FVector2f(GLumenIrradianceFieldShortRangeAORadius * 2.0f) / FVector2f(View.ViewRect.Size());
		FBlueNoise BlueNoise = GetBlueNoiseGlobalParameters();
		const uint32 DownsampleFactor = LumenShortRangeAO::GetRequestedDownsampleFactor();
		ScreenSpaceBentNormalParameters = ComputeScreenSpaceShortRangeAO(GraphBuilder, Scene, View, SceneTextures, LightingChannelsTexture, BlueNoise, MaxScreenTraceFraction.X, 1.0f, ComputePassFlags);
	}

	FRDGTextureDesc DiffuseIndirectDesc = FRDGTextureDesc::Create2DArray(DownsampledBufferSize, PF_FloatRGB, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV, 1);
	FRDGTextureRef DiffuseIndirect = GraphBuilder.CreateTexture(DiffuseIndirectDesc, TEXT("r.Lumen.IrradianceFieldGather.DiffuseIndirect"));

	FRDGTextureDesc RoughSpecularIndirectDesc = FRDGTextureDesc::Create2DArray(DownsampledBufferSize, PF_FloatRGB, FClearValueBinding::Black, TexCreate_ShaderResource | TexCreate_UAV, 1);
	FRDGTextureRef RoughSpecularIndirect = GraphBuilder.CreateTexture(RoughSpecularIndirectDesc, TEXT("r.Lumen.IrradianceFieldGather.RoughSpecularIndirect"));

	const bool bAdaptivePlacement = RadianceCacheInputs.UseAdaptiveProbes != 0;
	FRDGBufferRef ProbeAdaptiveIndices = bAdaptivePlacement ? GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeAdaptiveIndices) : nullptr;
	FRDGBufferRef ProbeInterpolationMisses = bAdaptivePlacement ? GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeInterpolationMisses) : nullptr;
	FRDGBufferRef ProbeLastUsedFrame = bAdaptivePlacement ? GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeLastUsedFrame) : nullptr;
	FRDGBufferRef ProbeValid = GraphBuilder.RegisterExternalBuffer(RadianceCacheState.ProbeValid);
	const bool bR8BufferSupported = UE::PixelFormat::HasCapabilities(PF_R8_UINT, EPixelFormatCapabilities::TypedUAVLoad) && UE::PixelFormat::HasCapabilities(PF_R8_UINT, EPixelFormatCapabilities::TypedUAVStore);

	if (RadianceCacheState.BlendedIrradianceAtlas)
	{
		RadianceCacheParameters.RadianceCacheFinalIrradianceAtlas = GraphBuilder.RegisterExternalTexture(RadianceCacheState.BlendedIrradianceAtlas);
		RadianceCacheParameters.RadianceCacheProbeOcclusionAtlas = GraphBuilder.RegisterExternalTexture(RadianceCacheState.BlendedOcclusionAtlas);
		FRDGBufferRef BlendedProbeWorldOffset = GraphBuilder.RegisterExternalBuffer(RadianceCacheState.BlendedProbeWorldOffset);
		RadianceCacheParameters.ProbeWorldOffset = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(BlendedProbeWorldOffset, RadianceCacheState.ProbeWorldOffsetFormat));
	}

	{
		FInterpolateIrradianceProbesToPixelsCS::FParameters* PassParameters = GraphBuilder.AllocParameters<FInterpolateIrradianceProbesToPixelsCS::FParameters>();
		PassParameters->RWDiffuseIndirect = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(DiffuseIndirect));
		PassParameters->RWRoughSpecularIndirect = GraphBuilder.CreateUAV(FRDGTextureUAVDesc(RoughSpecularIndirect));
		PassParameters->RadianceCacheParameters = RadianceCacheParameters;
		PassParameters->View = View.ViewUniformBuffer;
		PassParameters->SceneTexturesStruct = SceneTextures.UniformBuffer;
		PassParameters->Substrate = Substrate::BindSubstrateGlobalUniformParameters(View);
		PassParameters->ReflectionStruct = CreateReflectionUniformBuffer(GraphBuilder, View);
		LumenReflections::SetupCompositeParameters(View, PassParameters->ReflectionsCompositeParameters);
		PassParameters->UpsampleParameters = UpsampleParameters;
		PassParameters->ProbeAdaptiveIndices = bAdaptivePlacement ? GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ProbeAdaptiveIndices, PF_R32_UINT)) : nullptr;
		PassParameters->ProbeValid = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ProbeValid, bR8BufferSupported ? PF_R8_UINT : PF_R32_UINT));
		PassParameters->RWProbeLastUsedFrame = bAdaptivePlacement ? GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ProbeLastUsedFrame, PF_R32_UINT)) : nullptr;
		PassParameters->RWProbeInterpolationMisses = bAdaptivePlacement ? GraphBuilder.CreateUAV(FRDGBufferUAVDesc(ProbeInterpolationMisses, PF_R32_UINT)) : nullptr;
		const uint32 InterpolationFeedbackTileSize = FMath::RoundUpToPowerOfTwo(FMath::Clamp(GLumenIrradianceFieldAdaptivePlacementInterpolationFeedbackTileSize, 1, 256));
		PassParameters->InterpolationFeedbackTileWrapMask = InterpolationFeedbackTileSize - 1;
		PassParameters->InterpolationFeedbackTileJitter = GetFeedbackBufferTileJitter(InterpolationFeedbackTileSize, View.ViewState->GetFrameIndex());
		PassParameters->MaxNumProbes = MaxNumProbes;
		PassParameters->ProbeOcclusionBias = GLumenIrradianceFieldProbeOcclusionBias;
		PassParameters->MaxRoughnessToEvaluateRoughSpecular = GLumenIrradianceFieldMaxRoughnessToEvaluateRoughSpecular;

		FInterpolateIrradianceProbesToPixelsCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FInterpolateIrradianceProbesToPixelsCS::FInterpolateDownsampleFactor>(InterpolateDownsampleFactor);
		PermutationVector.Set<FInterpolateIrradianceProbesToPixelsCS::FAdaptiveProbes>(bAdaptivePlacement);
		auto ComputeShader = View.ShaderMap->GetShader<FInterpolateIrradianceProbesToPixelsCS>(PermutationVector);

		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("InterpolateIrradianceProbesToPixels DownsampleFactor=%u, %ux%u", InterpolateDownsampleFactor, UpsampleParameters.IntegrateViewSize.X, UpsampleParameters.IntegrateViewSize.Y),
			ComputePassFlags,
			ComputeShader,
			PassParameters,
			FComputeShaderUtils::GetGroupCount(UpsampleParameters.IntegrateViewSize, FInterpolateIrradianceProbesToPixelsCS::GetGroupSize()));
	}

	const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);

	FSSDSignalTextures DenoiserOutputs;
	DenoiserOutputs.Textures[0] = DiffuseIndirect;
	DenoiserOutputs.Textures[1] = SystemTextures.Black;
	DenoiserOutputs.Textures[2] = RoughSpecularIndirect;

	return DenoiserOutputs;
}

