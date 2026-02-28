"""
Enhanced Flight Data Preprocessing & Calibration
Produces route-wise, class-wise statistics for Multi-Class RL environment

OUTPUT FORMAT (route_stats.pkl):
{
  "Delhi-Mumbai": {
      "Economy": {
          "price_stats": {mean, median, std, q25, q75, ...},
          "competitor_prices": {airline1: price, airline2: price, ...},
          "competitor_details": {airline1: {median, mean, count, ...}, ...}
      },
      "Business": {
          "price_stats": {...},
          "competitor_prices": {...},
          "competitor_details": {...}
      }
  },
  ...
}
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path


class FlightDataProcessor:
    """
    Enhanced flight data processor with:
    - Multi-class support (Economy + Business)
    - Comprehensive competitor analysis
    - Statistical validation
    - Detailed reporting
    """
    
    def __init__(self, verbose=True):
        self.route_stats = {}
        self.verbose = verbose
        self.data_quality_report = {}

    # =================================================
    # DATA LOADING & VALIDATION
    # =================================================
    def load_data(self, filepath="data/flight_data.csv"):
        """Load and validate flight data"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"\n❌ Flight data not found at: {filepath}\n"
                f"Please ensure your CSV file is in the correct location.\n"
                f"Required columns: route, airline, price, class_category"
            )

        df = pd.read_csv(filepath)
        
        if self.verbose:
            print(f"\n✓ Loaded {len(df):,} flight records from {filepath}")

        # Validate required columns
        required_cols = ["route", "airline", "price", "class_category"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            # Try to auto-create route column
            if "route" in missing and "from" in df.columns and "to" in df.columns:
                df["route"] = df["from"] + "-" + df["to"]
                missing.remove("route")
                if self.verbose:
                    print(f"  ✓ Auto-created 'route' column from 'from' and 'to'")
        
        if missing:
            raise ValueError(
                f"❌ Missing required columns: {missing}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # Data cleaning
        df = self._clean_data(df)
        
        return df

    def _clean_data(self, df):
        """Clean and validate data"""
        
        initial_count = len(df)
        
        # Remove rows with missing critical values
        df = df.dropna(subset=["route", "airline", "price", "class_category"])
        
        # Remove invalid prices
        df = df[df["price"] > 0]
        
        # Standardize class names
        class_mapping = {
            "economy": "Economy",
            "ECONOMY": "Economy",
            "business": "Business",
            "BUSINESS": "Business",
            "first": "First",
            "FIRST": "First"
        }
        df["class_category"] = df["class_category"].replace(class_mapping)
        
        # Only keep Economy and Business (most common)
        df = df[df["class_category"].isin(["Economy", "Business"])]
        
        cleaned_count = len(df)
        removed = initial_count - cleaned_count
        
        if self.verbose and removed > 0:
            print(f"  ✓ Cleaned data: removed {removed} invalid records ({removed/initial_count*100:.1f}%)")
        
        return df

    # =================================================
    # MAIN ANALYSIS: ROUTE + CLASS
    # =================================================
    def analyze_route_by_class(self, df, route):
        """
        Comprehensive analysis of one route, separated by class
        Returns statistics for Economy and Business classes
        """
        
        route_df = df[df["route"] == route].copy()

        if route_df.empty:
            raise ValueError(f"No data for route: {route}")

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📊 ANALYZING ROUTE: {route}")
            print(f"{'='*80}")
            print(f"Total flights: {len(route_df)}")

        route_result = {}
        
        # Analyze each class separately
        for cls in ["Economy", "Business"]:
            cls_df = route_df[route_df["class_category"] == cls]
            
            if self.verbose:
                print(f"\n🎫 {cls} Class Analysis:")
                print(f"   Flights: {len(cls_df)}")

            # Skip if insufficient data
            if len(cls_df) < 5:
                if self.verbose:
                    print(f"   ⚠️  Insufficient data (need ≥5 flights, have {len(cls_df)})")
                continue

            # Compute statistics
            class_stats = self._compute_class_statistics(cls_df, cls)
            
            if class_stats:
                route_result[cls] = class_stats

        if not route_result:
            raise ValueError(
                f"❌ No valid class data for route {route}\n"
                f"   Economy flights: {len(route_df[route_df['class_category']=='Economy'])}\n"
                f"   Business flights: {len(route_df[route_df['class_category']=='Business'])}"
            )

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✓ Completed analysis for {route}")
            print(f"  Classes calibrated: {list(route_result.keys())}")
            print(f"{'='*80}")

        return route_result

    def _compute_class_statistics(self, cls_df, class_name):
        """Compute comprehensive statistics for a class"""
        
        # ------------------------------
        # PRICE STATISTICS
        # ------------------------------
        prices = cls_df["price"]
        
        price_stats = {
            "mean": float(prices.mean()),
            "median": float(prices.median()),
            "std": float(prices.std()),
            "q10": float(prices.quantile(0.10)),
            "q25": float(prices.quantile(0.25)),
            "q75": float(prices.quantile(0.75)),
            "q90": float(prices.quantile(0.90)),
            "min": float(prices.min()),
            "max": float(prices.max()),
            "count": int(len(cls_df)),
            "cv": float(prices.std() / prices.mean())  # Coefficient of variation
        }

        if self.verbose:
            print(f"   Price Statistics:")
            print(f"      Mean:   ₹{price_stats['mean']:,.0f}")
            print(f"      Median: ₹{price_stats['median']:,.0f}")
            print(f"      Std:    ₹{price_stats['std']:,.0f}")
            print(f"      Range:  ₹{price_stats['min']:,.0f} - ₹{price_stats['max']:,.0f}")
            print(f"      IQR:    ₹{price_stats['q25']:,.0f} - ₹{price_stats['q75']:,.0f}")

        # ------------------------------
        # COMPETITOR ANALYSIS
        # ------------------------------
        competitor_prices = {}
        competitor_details = {}
        
        airlines = cls_df["airline"].unique()
        
        if self.verbose:
            print(f"   Competitors: {len(airlines)} airlines")

        for airline in airlines:
            airline_df = cls_df[cls_df["airline"] == airline]
            
            if len(airline_df) == 0:
                continue
            
            airline_prices = airline_df["price"]
            
            # Use median (more robust to outliers)
            median_price = float(airline_prices.median())
            mean_price = float(airline_prices.mean())
            
            competitor_prices[airline] = median_price
            
            competitor_details[airline] = {
                "median": median_price,
                "mean": mean_price,
                "std": float(airline_prices.std()) if len(airline_df) > 1 else 0.0,
                "count": int(len(airline_df)),
                "min": float(airline_prices.min()),
                "max": float(airline_prices.max()),
                "market_share": float(len(airline_df) / len(cls_df))  # % of flights
            }
            
            if self.verbose:
                print(f"      {airline:20s}: ₹{median_price:7,.0f} "
                      f"(n={len(airline_df):3d}, share={competitor_details[airline]['market_share']*100:4.1f}%)")

        # Validate we have competitors
        if not competitor_prices:
            if self.verbose:
                print(f"   ⚠️  No valid competitor data")
            return None

        # ------------------------------
        # DEMAND INDICATORS (if available)
        # ------------------------------
        demand_indicators = self._compute_demand_indicators(cls_df)

        # ------------------------------
        # ASSEMBLE RESULTS
        # ------------------------------
        return {
            "price_stats": price_stats,
            "competitor_prices": competitor_prices,
            "competitor_details": competitor_details,
            "demand_indicators": demand_indicators,
            "data_quality": {
                "sample_size": len(cls_df),
                "airlines_count": len(airlines),
                "price_cv": price_stats["cv"],
                "min_competitor_flights": min([d["count"] for d in competitor_details.values()])
            }
        }

    def _compute_demand_indicators(self, cls_df):
        """Compute additional demand-related metrics if data available"""
        
        indicators = {}
        
        # Stops analysis (if available)
        if "stops" in cls_df.columns:
            stops_dist = cls_df["stops"].value_counts(normalize=True).to_dict()
            indicators["stops_distribution"] = {int(k): float(v) for k, v in stops_dist.items()}
        
        # Time of day analysis (if available)
        if "dep_hour" in cls_df.columns:
            # Categorize into periods
            def categorize_hour(hour):
                if 6 <= hour < 12:
                    return "Morning"
                elif 12 <= hour < 18:
                    return "Afternoon"
                elif 18 <= hour < 22:
                    return "Evening"
                else:
                    return "Night"
            
            cls_df["time_period"] = cls_df["dep_hour"].apply(categorize_hour)
            time_dist = cls_df["time_period"].value_counts(normalize=True).to_dict()
            indicators["time_distribution"] = time_dist
        
        # Duration analysis (if available)
        if "duration_in_min" in cls_df.columns:
            indicators["avg_duration_min"] = float(cls_df["duration_in_min"].mean())
        
        return indicators

    # =================================================
    # BATCH PROCESSING
    # =================================================
    def run_full_calibration(self, df, routes=None):
        """
        Run calibration for all routes (or specified routes)
        Returns comprehensive statistics for RL environment
        """
        
        if routes is None:
            routes = df["route"].unique()
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING FULL CALIBRATION")
        print(f"{'='*80}")
        print(f"Routes to process: {len(routes)}")
        print(f"Total records: {len(df):,}")
        
        # Class distribution
        class_dist = df["class_category"].value_counts()
        print(f"\nClass distribution:")
        for cls, count in class_dist.items():
            print(f"  {cls}: {count:,} flights ({count/len(df)*100:.1f}%)")

        success_count = 0
        fail_count = 0
        
        for i, route in enumerate(routes, 1):
            try:
                if self.verbose:
                    print(f"\n[{i}/{len(routes)}] Processing {route}...")
                
                self.route_stats[route] = self.analyze_route_by_class(df, route)
                success_count += 1
                
            except Exception as e:
                if self.verbose:
                    print(f"\n[{i}/{len(routes)}] ❌ Failed to process {route}: {e}")
                fail_count += 1
                continue

        # Summary
        print(f"\n{'='*80}")
        print(f"📈 CALIBRATION SUMMARY")
        print(f"{'='*80}")
        print(f"✓ Successfully calibrated: {success_count}/{len(routes)} routes")
        if fail_count > 0:
            print(f"❌ Failed: {fail_count} routes")
        print(f"\nCalibrated routes: {list(self.route_stats.keys())}")

        if not self.route_stats:
            raise RuntimeError("❌ No routes were successfully calibrated")

        # Generate quality report
        self._generate_quality_report()

        return self.route_stats

    def _generate_quality_report(self):
        """Generate data quality report"""
        
        print(f"\n{'='*80}")
        print(f"📊 DATA QUALITY REPORT")
        print(f"{'='*80}")
        
        for route, route_data in self.route_stats.items():
            print(f"\n{route}:")
            
            for cls, cls_data in route_data.items():
                quality = cls_data["data_quality"]
                print(f"  {cls}:")
                print(f"    Sample size: {quality['sample_size']} flights")
                print(f"    Airlines: {quality['airlines_count']}")
                print(f"    Price CV: {quality['price_cv']:.2f}")
                print(f"    Min competitor data: {quality['min_competitor_flights']} flights")
                
                # Quality assessment
                if quality['sample_size'] >= 20 and quality['airlines_count'] >= 3:
                    print(f"    Quality: ✓ EXCELLENT")
                elif quality['sample_size'] >= 10 and quality['airlines_count'] >= 2:
                    print(f"    Quality: ✓ Good")
                else:
                    print(f"    Quality: ⚠️  Limited (consider with caution)")

    # =================================================
    # SAVE / LOAD
    # =================================================
    def save_route_stats(self, filepath="data/route_stats.pkl"):
        """Save calibrated statistics to pickle file"""
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump(self.route_stats, f)
        
        file_size = os.path.getsize(filepath) / 1024  # KB
        
        print(f"\n✓ Saved route statistics to {filepath}")
        print(f"  File size: {file_size:.1f} KB")
        print(f"  Routes: {len(self.route_stats)}")

    def load_route_stats(self, filepath="data/route_stats.pkl"):
        """Load previously calibrated statistics"""
        
        if not os.path.exists(filepath):
            print(f"⚠️  No saved route statistics found at {filepath}")
            return False
        
        with open(filepath, "rb") as f:
            self.route_stats = pickle.load(f)
        
        print(f"✓ Loaded route statistics from {filepath}")
        print(f"  Routes: {len(self.route_stats)}")
        print(f"  Available routes: {list(self.route_stats.keys())}")
        
        return True

    # =================================================
    # VISUALIZATION & EXPORT
    # =================================================
    def export_summary(self, filepath="data/calibration_summary.txt"):
        """Export human-readable summary"""
        
        with open(filepath, "w") as f:
            f.write("="*80 + "\n")
            f.write("FLIGHT DATA CALIBRATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Routes: {len(self.route_stats)}\n\n")
            
            for route, route_data in self.route_stats.items():
                f.write(f"\n{route}\n")
                f.write("-"*80 + "\n")
                
                for cls, cls_data in route_data.items():
                    stats = cls_data["price_stats"]
                    comps = cls_data["competitor_prices"]
                    
                    f.write(f"\n{cls} Class:\n")
                    f.write(f"  Price: ₹{stats['median']:,.0f} (median), ₹{stats['mean']:,.0f} (mean)\n")
                    f.write(f"  Range: ₹{stats['min']:,.0f} - ₹{stats['max']:,.0f}\n")
                    f.write(f"  Sample: {stats['count']} flights\n")
                    f.write(f"  Competitors: {len(comps)}\n")
                    
                    for airline, price in sorted(comps.items(), key=lambda x: x[1]):
                        f.write(f"    {airline}: ₹{price:,.0f}\n")
        
        print(f"\n✓ Exported summary to {filepath}")

    def get_environment_params(self, route):
        """
        Get environment parameters for a specific route
        Useful for creating route-specific RL environments
        """
        
        if route not in self.route_stats:
            raise ValueError(f"Route {route} not found in calibrated data")
        
        route_data = self.route_stats[route]
        
        params = {
            "route": route,
            "classes": {}
        }
        
        for cls, cls_data in route_data.items():
            stats = cls_data["price_stats"]
            comps = cls_data["competitor_prices"]
            
            params["classes"][cls] = {
                "base_price": stats["median"],
                "price_mean": stats["mean"],
                "price_std": stats["std"],
                "price_min": stats["q25"],
                "price_max": stats["q75"] * 1.3,
                "competitor_prices": comps,
                "sample_size": stats["count"],
                "n_competitors": len(comps)
            }
        
        return params


# =================================================
# MAIN EXECUTION
# =================================================
if __name__ == "__main__":
    print("="*80)
    print("  ENHANCED FLIGHT DATA CALIBRATION")
    print("  Multi-Class Support (Economy + Business)")
    print("="*80)

    processor = FlightDataProcessor(verbose=True)

    try:
        # Load dataset
        df = processor.load_data("data/flight_data.csv")

        # Run full calibration
        processor.run_full_calibration(df)

        # Save results
        processor.save_route_stats("data/route_stats.pkl")
        processor.export_summary("data/calibration_summary.txt")

        print("\n" + "="*80)
        print("✓ CALIBRATION COMPLETE")
        print("="*80)
        
        print(f"\n📦 Output Files:")
        print(f"  • route_stats.pkl - Calibrated statistics for RL environment")
        print(f"  • calibration_summary.txt - Human-readable summary")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Use route_stats.pkl in your AirlineRevenueEnv")
        print(f"  2. Train your RL agent with calibrated parameters")
        print(f"  3. Evaluate on different routes")
        
        # Show sample
        sample_route = next(iter(processor.route_stats))
        print(f"\n📊 Sample Route: {sample_route}")
        sample_params = processor.get_environment_params(sample_route)
        print(f"  Classes: {list(sample_params['classes'].keys())}")
        
        for cls, params in sample_params['classes'].items():
            print(f"\n  {cls}:")
            print(f"    Base price: ₹{params['base_price']:,.0f}")
            print(f"    Range: ₹{params['price_min']:,.0f} - ₹{params['price_max']:,.0f}")
            print(f"    Competitors: {params['n_competitors']}")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\n💡 Solution:")
        print(f"  1. Place your flight_data.csv in the data/ folder")
        print(f"  2. Ensure required columns: route, airline, price, class_category")
        print(f"  3. Run this script again")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)