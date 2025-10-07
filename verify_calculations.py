# verify_calculations.py
import pandas as pd
import requests
from io import StringIO

def calculate_business_intelligence(csv_text):
    """Your exact function from the main script"""
    df = pd.read_csv(StringIO(csv_text))
    
    # Calculate comprehensive business intelligence
    daily_stats = df.groupby('day').agg({
        'total_bill': ['mean', 'sum', 'count'],
        'tip': ['mean', 'sum'],
        'size': 'mean'
    }).round(2)
    
    time_stats = df.groupby('time').agg({
        'total_bill': ['mean', 'sum'],
        'tip': ['mean', 'sum'],
        'size': 'mean'
    }).round(2)
    
    # Calculate actual percentages and business metrics
    df['tip_percentage'] = (df['tip'] / df['total_bill'] * 100)
    overall_tip_rate = df['tip_percentage'].mean()
    
    # Find best performing segments
    best_tip_day = df.groupby('day')['tip_percentage'].mean().idxmax()
    best_revenue_time = df.groupby('time')['total_bill'].sum().idxmax()
    
    return f"""
📊 CALCULATED BUSINESS INTELLIGENCE REPORT:

💰 FINANCIAL PERFORMANCE:
- Total Revenue: ${df['total_bill'].sum():.2f}
- Total Tips: ${df['tip'].sum():.2f}
- Overall Tip Rate: {overall_tip_rate:.1f}%
- Average Transaction: ${df['total_bill'].mean():.2f}

📅 DAILY BUSINESS INTELLIGENCE:
{chr(10).join([f"- {day}: ${daily_stats['total_bill']['mean'][day]:.2f} avg bill, {daily_stats['total_bill']['count'][day]} transactions, ${daily_stats['total_bill']['sum'][day]:.2f} total" for day in daily_stats.index])}

⏰ TIME-BASED PERFORMANCE:
{chr(10).join([f"- {time}: ${time_stats['total_bill']['mean'][time]:.2f} avg bill, ${time_stats['total_bill']['sum'][time]:.2f} total revenue" for time in time_stats.index])}

🎯 KEY BUSINESS INSIGHTS:
- Most Profitable Day: {best_tip_day} (highest tip percentage)
- Highest Revenue Time: {best_revenue_time}
- Customer Party Size: {df['size'].mean():.1f} average
- Busiest Day: {daily_stats['total_bill']['count'].idxmax()} ({daily_stats['total_bill']['count'].max()} transactions)

📈 PERFORMANCE METRICS:
- Weekend vs Weekday Revenue: ${df[df['day'].isin(['Sat','Sun'])]['total_bill'].sum():.2f} vs ${df[~df['day'].isin(['Sat','Sun'])]['total_bill'].sum():.2f}
- Dinner Premium: {((df[df['time']=='Dinner']['total_bill'].mean() - df[df['time']=='Lunch']['total_bill'].mean()) / df[df['time']=='Lunch']['total_bill'].mean() * 100):.1f}% higher bills
"""

def verify_manual_calculations():
    """Manually verify the key claims"""
    resp = requests.get("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
    df = pd.read_csv(StringIO(resp.text))
    
    print("🔍 MANUAL VERIFICATION OF CALCULATIONS")
    print("=" * 60)
    
    # Check the specific claims from your output
    print("💰 CHECKING REVENUE CLAIMS:")
    
    # Weekend vs Weekday (as defined in your function)
    weekend_revenue = df[df['day'].isin(['Sat','Sun'])]['total_bill'].sum()
    weekday_revenue = df[~df['day'].isin(['Sat','Sun'])]['total_bill'].sum()
    total_revenue = df['total_bill'].sum()
    
    print(f"Weekend (Sat+Sun) revenue: ${weekend_revenue:.2f}")
    print(f"Weekday (Thur+Fri) revenue: ${weekday_revenue:.2f}") 
    print(f"Total revenue: ${total_revenue:.2f}")
    print(f"Weekend % of total: {(weekend_revenue/total_revenue)*100:.1f}%")
    
    # Dinner vs Lunch
    dinner_revenue = df[df['time'] == 'Dinner']['total_bill'].sum()
    lunch_revenue = df[df['time'] == 'Lunch']['total_bill'].sum()
    
    print(f"\n🍽️ CHECKING TIME CLAIMS:")
    print(f"Dinner revenue: ${dinner_revenue:.2f}")
    print(f"Lunch revenue: ${lunch_revenue:.2f}")
    print(f"Dinner vs Lunch % increase: {((dinner_revenue - lunch_revenue)/lunch_revenue)*100:.1f}%")
    
    # Check individual days
    print(f"\n📅 INDIVIDUAL DAY REVENUE:")
    for day in df['day'].unique():
        day_revenue = df[df['day'] == day]['total_bill'].sum()
        print(f"- {day}: ${day_revenue:.2f}")

# Run both verifications
if __name__ == "__main__":
    print("ACTUAL CALCULATIONS FROM YOUR FUNCTION:")
    print("=" * 50)
    resp = requests.get("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
    csv_text = resp.text
    result = calculate_business_intelligence(csv_text)
    print(result)
    
    print("\n" + "=" * 80)
    verify_manual_calculations()
