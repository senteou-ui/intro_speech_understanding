
def next_birthday(date, birthdays):
    sorted_dates = sorted(birthdays.keys())
    
    for d in sorted_dates:
        if d > date:
            return d, birthdays[d]
            
    first_date = sorted_dates[0]
    return first_date, birthdays[first_date]