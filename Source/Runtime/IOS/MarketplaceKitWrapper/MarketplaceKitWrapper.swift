// Copyright Epic Games, Inc. All Rights Reserved.

import Foundation
#if canImport(MarketplaceKit)
import MarketplaceKit
#endif

@objc
enum AppDistributorType : Int
{
	case AppStore
	case TestFlight
	case Marketplace
	case Web
	case Other
	case NotAvailable
}

@objcMembers class AppDistributorWrapper : NSObject
{
	public static func getCurrent(CompletionHandler: @escaping (AppDistributorType, String) -> Void)
	{
		Task
		{
#if canImport(MarketplaceKit)
			let (Distributor, Name) = if #available(iOS 17.4, *)
			{
				try await AppDistributor.current.distributorType
			}
			else
			{
				(AppDistributorType.NotAvailable, String())
			}
#else
			let (Distributor, Name) = (AppDistributorType.NotAvailable, String());
#endif

			DispatchQueue.main.async
			{
				CompletionHandler(Distributor, Name)
			}
		}
	}
}

#if canImport(MarketplaceKit)
@available(iOS 17.4, *)
private extension AppDistributor
{
	var distributorType: (AppDistributorType, String)
	{
		switch self
		{
		case .appStore:					return (AppDistributorType.AppStore, String())
		case .testFlight:				return (AppDistributorType.TestFlight, String())
		case .marketplace(let Name):	return (AppDistributorType.Marketplace, Name)
		case .web:						return (AppDistributorType.Web, String())
		case .other:					return (AppDistributorType.Other, String())
		default:						return (AppDistributorType.Other, String())
		}
	}
}
#endif
