"""
Analyze Real Flight Data and Calibrate Environment
ENHANCED: Multi-Route, Multi-Class Analysis (Economy + Business)
Run this FIRST to understand your data and set up the environment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import FlightDataProcessor
import os

def main():
    print("="*80)
    print("  REAL FLIGHT DATA ANALYSIS & CALIBRATION")
    print("  Multi-Route, Multi-Class (Economy + Business)")
    print("="*80)
    
    # Check if data exists
    data_path = 'data/flight_data.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: No flight data found at {data_path}")
        print("\n📋 Required Steps:")
        print("   1. Add your CSV file to: data/flight_data.csv")
        print("   2. Ensure it has these columns:")
        print("      - airline (e.g., 'SpiceJet', 'IndiGo')")
        print("      - from (e.g., 'Delhi')")
        print("      - to (e.g., 'Mumbai')")
        print("      - route (e.g., 'Delhi-Mumbai') OR we'll create it")
        print("      - price (e.g., 5953)")
        print("      - class_category (e.g., 'Economy', 'Business')")
        print("   3. Optional columns: duration_in_min, stops, dep_hour, dep_period")
        print("\n💡 Your data format example:")
        print("   airline,from,to,price,class_category,duration_in_min,stops")
        print("   SpiceJet,Delhi,Mumbai,5953,Economy,130,0")
        print("   Vistara,Delhi,Mumbai,12500,Business,130,0")
        return
    
    # Load processor
    processor = FlightDataProcessor(verbose=True)
    
    try:
        # Load data
        print("\n📂 Loading flight data...")
        df = processor.load_data(data_path)
        
        # Show data info
        print(f"\n📊 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Columns: {', '.join(df.columns)}")
        
        # Show class distribution
        if 'class_category' in df.columns:
            class_dist = df['class_category'].value_counts()
            print(f"\n   Class Distribution:")
            for cls, count in class_dist.items():
                print(f"      {cls}: {count:,} flights ({count/len(df)*100:.1f}%)")
        
        # Check for important columns
        important_cols = ['stops', 'dep_period', 'dep_hour', 'duration_in_min']
        available_cols = [col for col in important_cols if col in df.columns]
        missing_cols = [col for col in important_cols if col not in df.columns]
        
        if available_cols:
            print(f"   ✓ Available for analysis: {', '.join(available_cols)}")
        if missing_cols:
            print(f"   ⚠️ Missing (optional): {', '.join(missing_cols)}")
        
        # Get available routes
        print("\n" + "-"*80)
        routes = df['route'].unique()
        
        if len(routes) == 0:
            print("\n❌ No routes found in data!")
            return
        
        # Show route summary
        print(f"\n🎯 Found {len(routes)} unique routes")
        print("\nTop routes by flight count:")
        route_counts = df['route'].value_counts().head(15)
        
        for i, (route, count) in enumerate(route_counts.items(), 1):
            route_df = df[df['route'] == route]
            avg_price = route_df['price'].mean()
            
            # Count classes for this route
            if 'class_category' in route_df.columns:
                class_counts = route_df['class_category'].value_counts()
                class_info = ", ".join([f"{cls}:{cnt}" for cls, cnt in class_counts.items()])
            else:
                class_info = "N/A"
            
            print(f"   {i:2d}. {route:30s} ({count:5d} flights, avg: ₹{avg_price:,.0f}, classes: {class_info})")
        
        # For now, analyze the route with most data
        selected_route = route_counts.index[0]
        print(f"\n✨ Auto-selecting route with most data: {selected_route}")
        
        print("\n" + "="*80)
        
        # MULTI-CLASS ANALYSIS of selected route
        print("\n🔍 Running Multi-Class Analysis...")
        route_stats = processor.analyze_route_by_class(df, selected_route)
        
        # Get calibrated parameters for display
        env_params = get_env_params_summary(route_stats, selected_route)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        create_multiclass_visualizations(df, selected_route, route_stats)
        
        # Summary and recommendations
        print("\n" + "="*80)
        print("  ✅ ROUTE-SPECIFIC ANALYSIS COMPLETE!")
        print("="*80)
        
        print_route_summary(selected_route, route_stats, env_params)
        
        # Now run FULL calibration for all routes
        print("\n" + "="*80)
        print("  🚀 RUNNING FULL MULTI-ROUTE CALIBRATION")
        print("="*80)
        
        all_route_stats = processor.run_full_calibration(df)
        
        # Save statistics
        processor.save_route_stats()
        processor.export_summary()
        
        # Final summary
        print("\n" + "="*80)
        print("  ✅ FULL CALIBRATION COMPLETE!")
        print("="*80)
        
        print(f"\n📈 Calibration Summary:")
        print(f"   ✓ Routes Calibrated: {len(all_route_stats)}")
        print(f"   ✓ Total Records Processed: {len(df):,}")
        
        # Count total classes across all routes
        total_classes = sum(len(route_data) for route_data in all_route_stats.values())
        print(f"   ✓ Total Route-Class Combinations: {total_classes}")
        
        print(f"\n📁 Output Files:")
        print(f"   ✓ data/route_stats.pkl - Full calibration data for RL environment")
        print(f"   ✓ data/calibration_summary.txt - Human-readable summary")
        print(f"   ✓ results/route_analysis_{selected_route.replace('/', '_')}.png - Detailed visualization")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. ✓ Route statistics saved and ready")
        print(f"   2. Run: python app.py")
        print(f"      (to start interactive dashboard)")
        print(f"   3. Run: python main.py --route '{selected_route}'")
        print(f"      (to train RL agent on specific route)")
        print(f"   4. Run: python main.py")
        print(f"      (to train on ALL {len(all_route_stats)} calibrated routes)")
        
        print("\n💡 The environment is now calibrated with YOUR real data!")
        print("   ✓ Multiple routes supported")
        print("   ✓ Multi-class pricing (Economy + Business)")
        print("   ✓ Realistic competitor behavior")
        print("   ✓ All parameters derived from actual flight data")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("   - Check your CSV format")
        print("   - Ensure 'class_category' column exists with values: 'Economy', 'Business'")
        print("   - Ensure price column has numeric values")
        print("   - Verify airline and route columns exist")
        print("   - Need at least 5 flights per route-class combination")


def get_env_params_summary(route_stats, route_name):
    """Extract key parameters for display"""
    params = {
        'route': route_name,
        'classes': {}
    }
    
    for cls, cls_data in route_stats.items():
        stats = cls_data['price_stats']
        comps = cls_data['competitor_prices']
        
        params['classes'][cls] = {
            'base_price': stats['median'],
            'mean_price': stats['mean'],
            'std': stats['std'],
            'range': (stats['min'], stats['max']),
            'iqr': (stats['q25'], stats['q75']),
            'competitors': len(comps),
            'sample_size': stats['count']
        }
    
    return params


def print_route_summary(route, route_stats, env_params):
    """Print detailed summary for the analyzed route"""
    
    print(f"\n📈 Key Insights for {route}:")
    print(f"   Classes Available: {list(route_stats.keys())}")
    
    for cls, cls_data in route_stats.items():
        stats = cls_data['price_stats']
        comps = cls_data['competitor_prices']
        
        print(f"\n   {cls} Class:")
        print(f"      Base Price (Median): ₹{stats['median']:,.0f}")
        print(f"      Mean Price: ₹{stats['mean']:,.0f}")
        print(f"      Std Dev: ₹{stats['std']:,.0f}")
        print(f"      Price Range: ₹{stats['min']:,.0f} - ₹{stats['max']:,.0f}")
        print(f"      IQR: ₹{stats['q25']:,.0f} - ₹{stats['q75']:,.0f}")
        print(f"      Sample Size: {stats['count']} flights")
        print(f"      Competitors: {len(comps)}")
        
        if comps:
            print(f"\n      🏢 Competitor Prices:")
            for airline, price in sorted(comps.items(), key=lambda x: x[1]):
                details = cls_data['competitor_details'][airline]
                market_share = details['market_share'] * 100
                print(f"         {airline:20s} → ₹{price:7,.0f} (n={details['count']:3d}, share={market_share:4.1f}%)")


def create_multiclass_visualizations(df, route, route_stats):
    """Create comprehensive visualizations for multi-class analysis"""
    route_df = df[df['route'] == route].copy()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set style
    sns.set_style('darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Determine grid size based on available data
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle(f'Multi-Class Flight Analysis: {route}', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Color palette for classes
    class_colors = {'Economy': '#3b82f6', 'Business': '#8b5cf6'}
    
    # ===== ROW 1: Price Distributions =====
    
    # 1. Overall Price Distribution by Class
    ax1 = fig.add_subplot(gs[0, 0])
    for cls in route_stats.keys():
        cls_df = route_df[route_df['class_category'] == cls]
        ax1.hist(cls_df['price'], bins=25, alpha=0.6, 
                label=cls, color=class_colors.get(cls, 'gray'), edgecolor='black')
        
        # Add mean/median lines
        mean_val = route_stats[cls]['price_stats']['mean']
        median_val = route_stats[cls]['price_stats']['median']
        ax1.axvline(mean_val, color=class_colors.get(cls, 'gray'), 
                   linestyle='--', linewidth=2, alpha=0.8)
        ax1.axvline(median_val, color=class_colors.get(cls, 'gray'), 
                   linestyle=':', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Price (₹)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Price Distribution by Class', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Boxplot Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    class_data = [route_df[route_df['class_category'] == cls]['price'].values 
                  for cls in route_stats.keys()]
    bp = ax2.boxplot(class_data, labels=list(route_stats.keys()), 
                     patch_artist=True, showmeans=True)
    
    for patch, cls in zip(bp['boxes'], route_stats.keys()):
        patch.set_facecolor(class_colors.get(cls, 'gray'))
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Price (₹)', fontsize=11, fontweight='bold')
    ax2.set_title('Price Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Statistics Table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('tight')
    ax3.axis('off')
    
    table_data = [['Class', 'Median', 'Mean', 'Std', 'Count']]
    for cls, cls_data in route_stats.items():
        stats = cls_data['price_stats']
        table_data.append([
            cls,
            f"₹{stats['median']:,.0f}",
            f"₹{stats['mean']:,.0f}",
            f"₹{stats['std']:,.0f}",
            f"{stats['count']}"
        ])
    
    table = ax3.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.22, 0.22, 0.22, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Price Statistics Summary', fontsize=13, fontweight='bold', pad=20)
    
    # ===== ROW 2: Competitor Analysis =====
    
    for col_idx, cls in enumerate(route_stats.keys()):
        ax = fig.add_subplot(gs[1, col_idx])
        
        comp_prices = route_stats[cls]['competitor_prices']
        comp_details = route_stats[cls]['competitor_details']
        
        airlines = list(comp_prices.keys())
        medians = [comp_prices[a] for a in airlines]
        counts = [comp_details[a]['count'] for a in airlines]
        
        # Sort by price
        sorted_data = sorted(zip(airlines, medians, counts), key=lambda x: x[1])
        airlines_sorted = [x[0] for x in sorted_data]
        medians_sorted = [x[1] for x in sorted_data]
        counts_sorted = [x[2] for x in sorted_data]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(airlines_sorted)))
        bars = ax.barh(airlines_sorted, medians_sorted, color=colors, alpha=0.8)
        
        ax.set_xlabel('Median Price (₹)', fontsize=11, fontweight='bold')
        ax.set_title(f'{cls} - Competitor Prices', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add labels
        for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                   f' ₹{medians_sorted[i]:,.0f} (n={count})',
                   va='center', fontsize=9)
    
    # Market Share pie chart in third column
    if len(route_stats) == 2:  # If we have both classes
        ax_share = fig.add_subplot(gs[1, 2])
        
        class_counts = [route_stats[cls]['price_stats']['count'] for cls in route_stats.keys()]
        colors_pie = [class_colors.get(cls, 'gray') for cls in route_stats.keys()]
        
        wedges, texts, autotexts = ax_share.pie(class_counts, labels=list(route_stats.keys()),
                                                 autopct='%1.1f%%', colors=colors_pie,
                                                 startangle=90, textprops={'fontsize': 11})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax_share.set_title('Flight Distribution by Class', fontsize=13, fontweight='bold')
    
    # ===== ROW 3: Additional Analysis =====
    
    # Price by Stops (if available)
    ax_stops = fig.add_subplot(gs[2, 0])
    if 'stops' in route_df.columns:
        for cls in route_stats.keys():
            cls_df = route_df[route_df['class_category'] == cls]
            stops_data = cls_df.groupby('stops')['price'].median()
            ax_stops.plot(stops_data.index, stops_data.values, 
                         marker='o', linewidth=2, markersize=8,
                         label=cls, color=class_colors.get(cls, 'gray'))
        
        ax_stops.set_xlabel('Number of Stops', fontsize=11, fontweight='bold')
        ax_stops.set_ylabel('Median Price (₹)', fontsize=11, fontweight='bold')
        ax_stops.set_title('Price by Stops', fontsize=13, fontweight='bold')
        ax_stops.legend()
        ax_stops.grid(True, alpha=0.3)
    else:
        ax_stops.text(0.5, 0.5, 'Stops data not available', 
                     ha='center', va='center', fontsize=12, transform=ax_stops.transAxes)
        ax_stops.set_title('Price by Stops', fontsize=13, fontweight='bold')
    
    # Price by Time (if available)
    ax_time = fig.add_subplot(gs[2, 1])
    if 'dep_period' in route_df.columns:
        time_comparison = []
        for cls in route_stats.keys():
            cls_df = route_df[route_df['class_category'] == cls]
            time_data = cls_df.groupby('dep_period')['price'].median().sort_values()
            time_comparison.append((cls, time_data))
        
        x = np.arange(len(time_comparison[0][1]))
        width = 0.35
        
        for i, (cls, data) in enumerate(time_comparison):
            offset = width * (i - len(time_comparison)/2 + 0.5)
            ax_time.bar(x + offset, data.values, width, 
                       label=cls, color=class_colors.get(cls, 'gray'), alpha=0.8)
        
        ax_time.set_xlabel('Departure Period', fontsize=11, fontweight='bold')
        ax_time.set_ylabel('Median Price (₹)', fontsize=11, fontweight='bold')
        ax_time.set_title('Price by Time of Day', fontsize=13, fontweight='bold')
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(time_comparison[0][1].index, rotation=45, ha='right')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3, axis='y')
    else:
        ax_time.text(0.5, 0.5, 'Time data not available', 
                    ha='center', va='center', fontsize=12, transform=ax_time.transAxes)
        ax_time.set_title('Price by Time of Day', fontsize=13, fontweight='bold')
    
    # Duration scatter (if available)
    ax_duration = fig.add_subplot(gs[2, 2])
    if 'duration_in_min' in route_df.columns:
        for cls in route_stats.keys():
            cls_df = route_df[route_df['class_category'] == cls]
            ax_duration.scatter(cls_df['duration_in_min'], cls_df['price'], 
                              alpha=0.4, s=30, label=cls,
                              color=class_colors.get(cls, 'gray'))
        
        ax_duration.set_xlabel('Duration (minutes)', fontsize=11, fontweight='bold')
        ax_duration.set_ylabel('Price (₹)', fontsize=11, fontweight='bold')
        ax_duration.set_title('Price vs Duration', fontsize=13, fontweight='bold')
        ax_duration.legend()
        ax_duration.grid(True, alpha=0.3)
    else:
        ax_duration.text(0.5, 0.5, 'Duration data not available', 
                        ha='center', va='center', fontsize=12, transform=ax_duration.transAxes)
        ax_duration.set_title('Price vs Duration', fontsize=13, fontweight='bold')
    
    # ===== ROW 4: Calibration Summary =====
    
    # Calibration text summary
    ax_summary = fig.add_subplot(gs[3, :2])
    
    summary_text = f"ROUTE: {route}\n\n"
    
    for cls, cls_data in route_stats.items():
        stats = cls_data['price_stats']
        comps = cls_data['competitor_prices']
        quality = cls_data['data_quality']
        
        summary_text += f"{cls.upper()} CLASS:\n"
        summary_text += f"  • Sample Size: {stats['count']} flights\n"
        summary_text += f"  • Price (Median): ₹{stats['median']:,.0f}\n"
        summary_text += f"  • Price (Mean): ₹{stats['mean']:,.0f}\n"
        summary_text += f"  • Std Dev: ₹{stats['std']:,.0f}\n"
        summary_text += f"  • Range: ₹{stats['min']:,.0f} - ₹{stats['max']:,.0f}\n"
        summary_text += f"  • IQR: ₹{stats['q25']:,.0f} - ₹{stats['q75']:,.0f}\n"
        summary_text += f"  • Competitors: {len(comps)}\n"
        summary_text += f"  • Price Floor: ₹{stats['q25']:,.0f}\n"
        summary_text += f"  • Price Ceiling: ₹{stats['q75'] * 1.3:,.0f}\n\n"
    
    ax_summary.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
                   family='monospace', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax_summary.axis('off')
    ax_summary.set_title('Calibration Summary', fontsize=13, fontweight='bold', pad=10)
    
    # Competitor details table
    ax_comp_table = fig.add_subplot(gs[3, 2])
    ax_comp_table.axis('tight')
    ax_comp_table.axis('off')
    
    # Combine all competitors from all classes
    table_data = [['Class', 'Airline', 'Median', 'Count']]
    for cls in route_stats.keys():
        comp_prices = route_stats[cls]['competitor_prices']
        comp_details = route_stats[cls]['competitor_details']
        
        for airline in sorted(comp_prices.keys(), key=lambda x: comp_prices[x]):
            details = comp_details[airline]
            table_data.append([
                cls[:4],  # Abbreviate
                airline[:15],  # Truncate
                f"₹{details['median']:,.0f}",
                f"{details['count']}"
            ])
    
    comp_table = ax_comp_table.table(cellText=table_data, cellLoc='left', loc='center',
                                     colWidths=[0.15, 0.45, 0.25, 0.15])
    comp_table.auto_set_font_size(False)
    comp_table.set_fontsize(8)
    comp_table.scale(1, 1.8)
    
    # Style header
    for i in range(4):
        comp_table[(0, i)].set_facecolor('#40466e')
        comp_table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_comp_table.set_title('All Competitors', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    safe_route = route.replace("/", "_").replace("\\", "_").replace(" ", "_")
    save_path = f'results/route_analysis_{safe_route}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved visualization: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    main()