// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreTypes.h"
#include "RenderGraphFwd.h"
#include "Lumen/LumenFrontLayerTranslucency.h"

class FViewInfo;
class FSceneViewState;
struct FMinimalSceneTextures;
enum class EReflectionsMethod;

namespace StochasticLighting
{
	enum class EMaterialSource
	{
		GBuffer,
		FrontLayerGBuffer,
		HairStrands,
		MAX
	};

	enum class EStochasticSampleOffset
	{
		None,
		DownsampleFactor2x1,
		DownsampleFactor2x2,
		Both,
		MAX
	};

	int32 GetStateFrameIndex(const FSceneViewState* ViewState);

	bool IsStateFrameIndexOverridden();

	struct FRunConfig
	{
		ERDGPassFlags ComputePassFlags = ERDGPassFlags::Compute;
		int32 StateFrameIndexOverride = -1;
		bool bSubstrateOverflow = false;
		bool bCopyDepthAndNormal = false;
		bool bDownsampleDepthAndNormal = false;
		bool bTileClassifyLumen = false;
		bool bTileClassifyMegaLights = false;
		bool bTileClassifySubstrate = false;
		bool bReprojectLumen = false;
		bool bReprojectMegaLights = false;
	};

	struct FContext
	{
		FRDGBuilder& GraphBuilder;
		const FMinimalSceneTextures& SceneTextures;
		const FLumenFrontLayerTranslucencyGBufferParameters& FrontLayerTranslucencyGBuffer;
		StochasticLighting::EMaterialSource MaterialSource;
		FRDGTextureUAVRef DepthHistoryUAV = nullptr;
		FRDGTextureUAVRef NormalHistoryUAV = nullptr;
		FRDGTextureUAVRef DownsampledSceneDepthUAV = nullptr;
		FRDGTextureUAVRef DownsampledWorldNormalUAV = nullptr;
		FRDGTextureUAVRef LumenTileBitmaskUAV = nullptr;
		FRDGTextureUAVRef MegaLightsTileBitmaskUAV = nullptr;
		FRDGTextureUAVRef EncodedReprojectionVectorUAV = nullptr;
		FRDGTextureUAVRef LumenPackedPixelDataUAV = nullptr;
		FRDGTextureUAVRef MegaLightsPackedPixelDataUAV = nullptr;

		FContext(
			FRDGBuilder& InGraphBuilder,
			const FMinimalSceneTextures& InSceneTextures,
			const FLumenFrontLayerTranslucencyGBufferParameters& InFrontLayerTranslucencyGBuffer,
			StochasticLighting::EMaterialSource InMaterialSource);

		void Validate(const FRunConfig& RunConfig) const;

		void Run(const FViewInfo& View, EReflectionsMethod ViewReflectionsMethod, const FRunConfig& RunConfig);
	};
}