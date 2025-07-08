from datetime import datetime, timedelta
import investpy  # or use `import investpy_reborn as investpy` if you're using the fork

def get_weekly_events():
    monday = datetime.today() - timedelta(days=datetime.today().weekday())
    friday = monday + timedelta(days=4)

    from_date = monday.strftime('%d/%m/%Y')
    to_date = friday.strftime('%d/%m/%Y')

    return investpy.news.economic_calendar(
        importance=['high'],
        from_date=from_date,
        to_date=to_date,
        time_zone=None
    )

# Fetch and print high-impact events for this week
weekly_events = get_weekly_events()
print(weekly_events[['date', 'time', 'currency', 'importance', 'event', 'actual', 'forecast', 'previous']])
