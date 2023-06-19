from utils import (
    check_type,
    months_between,
    convert_datetime_to_date,
    outside_threshold
)

import datetime as dt
import pytest

def test_wrong_type():
    with pytest.raises(ValueError):
        check_type("awrongtype")

class TestMonthsBetween:
    def test_months_between_with_many_months(self):
        from_date = dt.date.fromisoformat("2020-01-14")
        to_date = dt.date.fromisoformat("2020-05-29")
        
        months = list(months_between(from_date, to_date))
        
        expected_values = [
            dt.date(2020, 1, 1),
            dt.date(2020, 2, 1),
            dt.date(2020, 3, 1),
            dt.date(2020, 4, 1),
            dt.date(2020, 5, 1)
        ]
        
        assert expected_values == months
        
    def test_months_between_with_same_date(self):
        datevalue = dt.date(2020, 6, 19)
        
        assert list(months_between(datevalue, datevalue)) == [dt.date(2020, 6, 1)]
        
    def test_months_with_wrong_dates(self):
        from_date = dt.date(2020, 5, 29)
        to_date = dt.date(2020, 1, 14)
        
        with pytest.raises(ValueError):
            list(months_between(from_date, to_date))
            
def test_convert_datetime_to_date():
    assert convert_datetime_to_date("2020-03-19 11:29:32") == dt.date(2020, 3, 19)
    

def test_outside_threshold_with_valid_date():
    item_date = dt.date(2020, 3, 19)
    threshold_date = dt.date(2020, 3, 1)
    
    assert not outside_threshold(item_date, threshold_date)

def test_outside_threshold_with_invalid_date():
    item_date = dt.date(2020, 3, 19)
    threshold_date = dt.date(2020, 5, 1)
    
    assert outside_threshold(item_date, threshold_date)
