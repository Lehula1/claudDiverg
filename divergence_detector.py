import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DIVERGENCE DETECTION (Your existing code)
# ============================================================================

def calculate_stochastic(df, period=14, smooth=3):
    """Calculate Stochastic %K with smoothing"""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    stoch_raw = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_k = stoch_raw.rolling(window=smooth).mean()
    return stoch_k

def find_pivots(series, lbL=3, lbR=2):
    """Find pivots: value must be highest/lowest in window [i-lbL, i+lbR]"""
    pivots_high = []
    pivots_low = []
    values = series.values

    for i in range(lbL, len(values) - lbR):
        window = values[i - lbL:i + lbR + 1]
        val = values[i]

        if val == np.max(window) and np.sum(window == val) == 1:
            pivots_high.append(i)

        if val == np.min(window) and np.sum(window == val) == 1:
            pivots_low.append(i)

    return np.array(pivots_high), np.array(pivots_low)

def detect_divergences(df, lbL=3, lbR=2, range_min=4, range_max=60,
                       min_diff=7, oversold=20, overbought=80):
    """Detect divergences using EXACT PineScript logic"""
    print("  Calculating stochastics...")
    df['k14'] = calculate_stochastic(df, period=14, smooth=3)
    df['k60'] = calculate_stochastic(df, period=60, smooth=10)
    df = df.dropna().copy()

    print("  Finding stochastic pivots...")
    stoch_highs, stoch_lows = find_pivots(df['k14'], lbL, lbR)

    # DEBUG: Show sample of detected pivot values
    print(f"    DEBUG: Sample of first 10 pivot highs:")
    for i in range(min(10, len(stoch_highs))):
        idx = stoch_highs[i]
        val = df['k14'].iloc[idx]
        print(f"      Index {idx}: K14 = {val:.1f}")

    bullish_divs = []
    bearish_divs = []

    # BULLISH DIVERGENCES
    print(f"  Scanning {len(stoch_lows)} consecutive stoch lows...")
    for i in range(1, len(stoch_lows)):
        curr_stoch_idx = stoch_lows[i]
        prev_stoch_idx = stoch_lows[i-1]
        bars_between = curr_stoch_idx - prev_stoch_idx

        if not (range_min <= bars_between <= range_max):
            continue

        curr_stoch = df['k14'].iloc[curr_stoch_idx]
        prev_stoch = df['k14'].iloc[prev_stoch_idx]

        k60_val = df['k60'].iloc[curr_stoch_idx]

        if curr_stoch > 60 or prev_stoch > 60:
            continue

        # STRICTER: Previous stoch must be deeply oversold (< 20)
        if prev_stoch >= 20:
            continue

        if not (curr_stoch > prev_stoch and
                (curr_stoch - prev_stoch) > min_diff and
                k60_val <= oversold):
            continue

        # Find actual price lows in a 10-bar window
        window_size = 10
        curr_search_start = max(0, curr_stoch_idx - window_size)
        curr_search_end = min(len(df), curr_stoch_idx + window_size + 1)
        prev_search_start = max(0, prev_stoch_idx - window_size)
        prev_search_end = min(len(df), prev_stoch_idx + window_size + 1)

        curr_low_window = df['low'].iloc[curr_search_start:curr_search_end]
        prev_low_window = df['low'].iloc[prev_search_start:prev_search_end]

        curr_3bar_low = curr_low_window.min()
        prev_3bar_low = prev_low_window.min()

        curr_price_idx = curr_low_window.idxmin()
        prev_price_idx = prev_low_window.idxmin()

        curr_price_idx_pos = df.index.get_loc(curr_price_idx)
        prev_price_idx_pos = df.index.get_loc(prev_price_idx)

        if abs(curr_price_idx_pos - curr_stoch_idx) > 3:
            continue
        if abs(prev_price_idx_pos - prev_stoch_idx) > 3:
            continue

        if curr_3bar_low <= prev_3bar_low:
            bullish_divs.append({
                'type': 'bullish',
                'date': df.index[curr_stoch_idx],
                'prev_price': prev_3bar_low,
                'curr_price': curr_3bar_low,
                'prev_stoch': prev_stoch,
                'curr_stoch': curr_stoch,
                'k60': k60_val,
                'price_idx_prev': prev_price_idx_pos,
                'price_idx_curr': curr_price_idx_pos,
                'stoch_idx_prev': prev_stoch_idx,
                'stoch_idx_curr': curr_stoch_idx,
                'strength': curr_stoch - prev_stoch,
                'bars_between': bars_between,
                'volume_curr': df['volume'].iloc[curr_stoch_idx],
                'volume_prev': df['volume'].iloc[prev_stoch_idx]
            })

    # BEARISH DIVERGENCES
    print(f"  Scanning {len(stoch_highs)} consecutive stoch highs...")
    for i in range(1, len(stoch_highs)):
        curr_stoch_idx = stoch_highs[i]
        prev_stoch_idx = stoch_highs[i-1]
        bars_between = curr_stoch_idx - prev_stoch_idx

        if not (range_min <= bars_between <= range_max):
            continue

        # Get the stoch values FIRST (need them for the intermediate check)
        curr_stoch = df['k14'].iloc[curr_stoch_idx]
        prev_stoch = df['k14'].iloc[prev_stoch_idx]

        k60_val = df['k60'].iloc[curr_stoch_idx]

        if curr_stoch < 40 or prev_stoch < 40:
            continue

        # STRICTER: Previous stoch must be strongly overbought (> 75)
        if prev_stoch <= 75:
            continue

        if not (curr_stoch < prev_stoch and
                (prev_stoch - curr_stoch) > min_diff and
                k60_val >= overbought):
            continue

        window_size = 10
        curr_search_start = max(0, curr_stoch_idx - window_size)
        curr_search_end = min(len(df), curr_stoch_idx + window_size + 1)
        prev_search_start = max(0, prev_stoch_idx - window_size)
        prev_search_end = min(len(df), prev_stoch_idx + window_size + 1)

        curr_high_window = df['high'].iloc[curr_search_start:curr_search_end]
        prev_high_window = df['high'].iloc[prev_search_start:prev_search_end]

        curr_3bar_high = curr_high_window.max()
        prev_3bar_high = prev_high_window.max()

        curr_price_idx = curr_high_window.idxmax()
        prev_price_idx = prev_high_window.idxmax()

        curr_price_idx_pos = df.index.get_loc(curr_price_idx)
        prev_price_idx_pos = df.index.get_loc(prev_price_idx)

        if abs(curr_price_idx_pos - curr_stoch_idx) > 3:
            continue
        if abs(prev_price_idx_pos - prev_stoch_idx) > 3:
            continue

        if curr_3bar_high >= prev_3bar_high:
            bearish_divs.append({
                'type': 'bearish',
                'date': df.index[curr_stoch_idx],
                'prev_price': prev_3bar_high,
                'curr_price': curr_3bar_high,
                'prev_stoch': prev_stoch,
                'curr_stoch': curr_stoch,
                'k60': k60_val,
                'price_idx_prev': prev_price_idx_pos,
                'price_idx_curr': curr_price_idx_pos,
                'stoch_idx_prev': prev_stoch_idx,
                'stoch_idx_curr': curr_stoch_idx,
                'strength': prev_stoch - curr_stoch,
                'bars_between': bars_between,
                'volume_curr': df['volume'].iloc[curr_stoch_idx],
                'volume_prev': df['volume'].iloc[prev_stoch_idx]
            })

    return bullish_divs, bearish_divs, df

# ============================================================================
# PART 2: SUCCESS ANALYSIS
# ============================================================================

def analyze_divergence_outcome(row, df, lookforward_bars=20):
    """
    Analyze what happened after a divergence signal

    For BULLISH: Success = price goes UP at least 0.3% before stopping out
    For BEARISH: Success = price goes DOWN at least 0.3% before stopping out

    Returns: (success, max_profit_pct, bars_to_peak, stopped_out)
    """
    div_type = row['type']
    div_date = pd.to_datetime(row['date'])
    entry_price = row['curr_price']

    try:
        div_idx = df.index.get_loc(div_date)
    except:
        return None, None, None, None

    future_data = df.iloc[div_idx+1:div_idx+1+lookforward_bars]

    if len(future_data) < 5:
        return None, None, None, None

    if div_type == 'bullish':
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()

        max_profit_pct = ((max_price - entry_price) / entry_price) * 100
        max_loss_pct = ((min_price - entry_price) / entry_price) * 100

        # Success: Price goes up at least 0.3% before going down 0.15%
        success = max_profit_pct >= 0.3 and max_loss_pct > -0.15
        stopped_out = max_loss_pct <= -0.15

        bars_to_peak = future_data['high'].idxmax()

    else:  # bearish
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()

        max_profit_pct = ((entry_price - min_price) / entry_price) * 100
        max_loss_pct = ((max_price - entry_price) / entry_price) * 100

        # Success: Price goes down at least 0.3% before going up 0.15%
        success = max_profit_pct >= 0.3 and max_loss_pct < 0.15
        stopped_out = max_loss_pct >= 0.15

        bars_to_peak = future_data['low'].idxmin()

    return success, max_profit_pct, bars_to_peak, stopped_out

def plot_divergence_with_outcome(row, df, success_label, chart_num, lookforward=20):
    """Plot divergence showing what happened after the signal"""

    div_date = pd.to_datetime(row['date'])
    div_type = row['type']

    try:
        div_idx = df.index.get_loc(div_date)
    except:
        print(f"  ⚠️  Could not find date {div_date} in dataframe")
        return

    # Get window: 60 bars before (to see prev pivot), 20 bars after
    start_idx = max(0, div_idx - 60)
    end_idx = min(len(df), div_idx + lookforward + 1)
    window_data = df.iloc[start_idx:end_idx].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=80)

    success_text = "✓ SUCCESS" if row['success'] else "✗ FAILED"
    color_text = 'green' if row['success'] else 'red'

    # Convert to Pacific time for display
    div_date_pt = div_date.tz_localize('UTC').tz_convert('US/Pacific')

    fig.suptitle(
        f"{success_label} - {div_type.upper()} Divergence\n"
        f"Date: {div_date_pt.strftime('%Y-%m-%d %H:%M PT')} | "
        f"Strength: {row['strength']:.1f} | "
        f"Outcome: {row['max_profit_pct']:.2f}% | "
        f"{success_text}",
        fontsize=14, fontweight='bold'
    )

    # PANEL 1: Price with candlesticks and outcome
    ax1 = axes[0]

    for i in range(len(window_data)):
        o = window_data['open'].iloc[i]
        h = window_data['high'].iloc[i]
        l = window_data['low'].iloc[i]
        c = window_data['close'].iloc[i]

        color = 'green' if c >= o else 'red'
        ax1.plot([i, i], [l, h], color='black', linewidth=0.8)

        body_height = abs(c - o)
        if body_height == 0:
            body_height = (h - l) * 0.05
        body_bottom = min(o, c)

        rect = plt.Rectangle((i-0.4, body_bottom), 0.8, body_height,
                            facecolor=color, edgecolor='black', linewidth=0.6, alpha=0.8)
        ax1.add_patch(rect)

    # Mark divergence point
    div_loc = div_idx - start_idx
    ax1.scatter([div_loc], [row['curr_price']], color='yellow', s=250,
               edgecolors='red', linewidths=3, zorder=10, marker='*', label='Entry Signal')

    # Vertical line at signal
    ax1.axvline(x=div_loc, color='orange', linestyle='--', alpha=0.6, linewidth=2.5)

    # Shade the outcome period
    outcome_color = 'lightgreen' if row['success'] else 'lightcoral'
    ax1.axvspan(div_loc, len(window_data)-1, alpha=0.15, color=outcome_color)

    # Mark the peak/trough if successful
    if row['success'] and row['bars_to_peak'] is not None:
        try:
            peak_idx = window_data.index.get_loc(row['bars_to_peak'])
            if div_type == 'bullish':
                peak_price = window_data.loc[row['bars_to_peak'], 'high']
                marker_color = 'lime'
                label_text = f'Target: +{row["max_profit_pct"]:.2f}%'
            else:
                peak_price = window_data.loc[row['bars_to_peak'], 'low']
                marker_color = 'lime'
                label_text = f'Target: +{row["max_profit_pct"]:.2f}%'

            ax1.scatter([peak_idx], [peak_price], color=marker_color, s=250,
                       edgecolors='darkgreen', linewidths=3, zorder=10, marker='*',
                       label=label_text)

            # Arrow from entry to target
            ax1.annotate('', xy=(peak_idx, peak_price),
                        xytext=(div_loc, row['curr_price']),
                        arrowprops=dict(arrowstyle='->', color='green', lw=3, alpha=0.7))
        except:
            pass

    ax1.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
    ax1.set_title('Bitcoin Price - What Happened After Signal', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Outcome box
    outcome_text = (
        f"Entry: ${row['curr_price']:.2f}\n"
        f"Max Profit: {row['max_profit_pct']:.2f}%\n"
        f"Result: {success_text}"
    )
    ax1.text(0.98, 0.98, outcome_text, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor=outcome_color, alpha=0.8, edgecolor=color_text, linewidth=2),
            verticalalignment='top', horizontalalignment='right',
            fontweight='bold', fontsize=10)

    # PANEL 2: Stochastic K14
    ax2 = axes[1]
    ax2.plot(range(len(window_data)), window_data['k14'], 'b-', linewidth=2, label='K14')
    ax2.axhline(y=80, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=60, color='orange', linestyle=':', alpha=0.4)
    ax2.axhline(y=40, color='purple', linestyle=':', alpha=0.4)

    # Mark stochastic pivots
    prev_stoch_loc = row['stoch_idx_prev'] - start_idx
    curr_stoch_loc = row['stoch_idx_curr'] - start_idx

    # Always plot current pivot (should always be visible)
    if 0 <= curr_stoch_loc < len(window_data):
        ax2.scatter([curr_stoch_loc], [row['curr_stoch']],
                   color='red', s=120, zorder=5, edgecolors='darkred', linewidths=2)

    # Plot previous pivot if visible, otherwise show it at edge with annotation
    if 0 <= prev_stoch_loc < len(window_data):
        ax2.scatter([prev_stoch_loc], [row['prev_stoch']],
                   color='red', s=120, zorder=5, edgecolors='darkred', linewidths=2)
        ax2.plot([prev_stoch_loc, curr_stoch_loc],
                [row['prev_stoch'], row['curr_stoch']],
                'r--', linewidth=2.5, alpha=0.7)
    else:
        # Previous pivot is outside window - show annotation
        ax2.annotate(f'Prev pivot: {row["prev_stoch"]:.1f}\n({abs(prev_stoch_loc)} bars back)',
                    xy=(0, row['prev_stoch']), xytext=(5, row['prev_stoch']),
                    fontsize=8, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax2.axvline(x=div_loc, color='orange', linestyle='--', alpha=0.6, linewidth=2.5)
    ax2.axvspan(div_loc, len(window_data)-1, alpha=0.15, color=outcome_color)

    ax2.set_ylabel('Stochastic K14', fontweight='bold', fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # PANEL 3: Stochastic K60
    ax3 = axes[2]
    ax3.plot(range(len(window_data)), window_data['k60'], 'purple', linewidth=2, label='K60 (Trend Filter)')
    ax3.axhline(y=80, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=20, color='g', linestyle='--', alpha=0.5, linewidth=1)

    ax3.scatter([div_loc], [row['k60']], color='red', s=120, zorder=5,
               edgecolors='darkred', linewidths=2)

    ax3.axvline(x=div_loc, color='orange', linestyle='--', alpha=0.6, linewidth=2.5)
    ax3.axvspan(div_loc, len(window_data)-1, alpha=0.15, color=outcome_color)

    ax3.set_ylabel('Stochastic K60', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Time (Pacific) | Bar Number', fontweight='bold', fontsize=11)
    ax3.set_ylim(-5, 105)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Format x-axis with Pacific time AND bar numbers
    step = max(1, len(window_data) // 10)
    positions = list(range(0, len(window_data), step))

    # Convert UTC times to Pacific
    labels = []
    for p in positions:
        if p < len(window_data):
            timestamp = window_data.index[p]
            # Convert to Pacific time
            pt_time = timestamp.tz_localize('UTC').tz_convert('US/Pacific')
            time_str = pt_time.strftime('%H:%M')
            label = f"{time_str}\nBar {p}"
            labels.append(label)
        else:
            labels.append('')

    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, rotation=0, ha='center', fontsize=8)

    # Add info box at bottom
    vol_ratio = row['volume_curr'] / row['volume_prev'] if row['volume_prev'] > 0 else 0
    info = (f"Strength: {row['strength']:.1f} | Bars Between: {row['bars_between']} | "
           f"K60: {row['k60']:.1f} | Volume Ratio: {vol_ratio:.2f}x")
    fig.text(0.5, 0.01, info, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    filename = f"{success_label.replace(' ', '_')}_{div_type}_{chart_num}.png"
    plt.savefig(filename, dpi=80, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()  # Display the chart
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    filename = 'BTC_USD_1min_20231222_20251221.csv'

    print("="*80)
    print("BITCOIN DIVERGENCE DETECTOR + SUCCESS ANALYZER")
    print("="*80)
    print(f"\nStep 1: Loading data from {filename}")

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    if df.index.duplicated().any():
        print(f"⚠️  Removing {df.index.duplicated().sum()} duplicate timestamps...")
        df = df[~df.index.duplicated(keep='first')]

    print(f"✓ Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")

    print("\nStep 2: Detecting divergences...")
    start_time = time.time()
    bullish, bearish, df = detect_divergences(df)

    # Filter weak divergences
    min_price_change_pct = 0.05
    bullish = [d for d in bullish if abs(d['curr_price'] - d['prev_price']) / d['prev_price'] * 100 >= min_price_change_pct]
    bearish = [d for d in bearish if abs(d['curr_price'] - d['prev_price']) / d['prev_price'] * 100 >= min_price_change_pct]

    print(f"✓ Found {len(bullish)} bullish and {len(bearish)} bearish divergences ({time.time()-start_time:.1f}s)")

    # Combine and save
    all_divs = bullish + bearish
    divs_df = pd.DataFrame(all_divs)
    divs_df.to_csv('divergences_found.csv', index=False)
    print(f"✓ Saved to divergences_found.csv")

    print("\nStep 3: Analyzing outcomes (what happened after each signal)...")
    results = []
    for idx, row in divs_df.iterrows():
        success, profit, peak, stopped = analyze_divergence_outcome(row, df)
        if success is not None:
            results.append({
                **row.to_dict(),
                'success': success,
                'max_profit_pct': profit,
                'bars_to_peak': peak,
                'stopped_out': stopped
            })

    results_df = pd.DataFrame(results)

    # Statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    successful = results_df[results_df['success'] == True]
    failed = results_df[results_df['success'] == False]

    print(f"\nTotal Analyzed: {len(results_df)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results_df)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results_df)*100:.1f}%)")

    bull_df = results_df[results_df['type'] == 'bullish']
    bear_df = results_df[results_df['type'] == 'bearish']

    print(f"\nBullish Divergences:")
    bull_success = bull_df[bull_df['success'] == True]
    print(f"  Success Rate: {len(bull_success)}/{len(bull_df)} ({len(bull_success)/len(bull_df)*100:.1f}%)")
    if len(bull_success) > 0:
        print(f"  Avg Profit: {bull_success['max_profit_pct'].mean():.2f}%")

    print(f"\nBearish Divergences:")
    bear_success = bear_df[bear_df['success'] == True]
    print(f"  Success Rate: {len(bear_success)}/{len(bear_df)} ({len(bear_success)/len(bear_df)*100:.1f}%)")
    if len(bear_success) > 0:
        print(f"  Avg Profit: {bear_success['max_profit_pct'].mean():.2f}%")

    results_df.to_csv('divergence_analysis.csv', index=False)
    print(f"\n✓ Saved full analysis to divergence_analysis.csv")

    # Create example charts
    print("\n" + "="*80)
    print("CREATING EXAMPLE CHARTS")
    print("="*80)

    # Get best and worst examples
    print("\n📊 Top 5 SUCCESSFUL divergences:")
    best_divs = results_df[results_df['success'] == True].nlargest(5, 'max_profit_pct')
    for i, (_, row) in enumerate(best_divs.iterrows(), 1):
        plot_divergence_with_outcome(row, df, f"GOOD_Example_{i}", i)

    print("\n📊 Top 5 FAILED divergences:")
    worst_divs = results_df[results_df['success'] == False].nsmallest(5, 'max_profit_pct')
    for i, (_, row) in enumerate(worst_divs.iterrows(), 1):
        plot_divergence_with_outcome(row, df, f"BAD_Example_{i}", i)

    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  • divergences_found.csv - All detected divergences")
    print("  • divergence_analysis.csv - Success/failure analysis")
    print("  • 10 PNG charts showing best/worst examples")
    print("\nKey Findings:")
    print(f"  • Overall Win Rate: {len(successful)/len(results_df)*100:.1f}%")
    print(f"  • Avg Profit on Winners: {successful['max_profit_pct'].mean():.2f}%")
    print(f"  • Bullish Win Rate: {len(bull_success)/len(bull_df)*100:.1f}%")
    print(f"  • Bearish Win Rate: {len(bear_success)/len(bear_df)*100:.1f}%")
    print("\nNext Step: Review the charts to verify the training criteria is correct!")
    print("="*80)
