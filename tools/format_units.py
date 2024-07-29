import json

def format_units(value, decimals = 18):
    """Divides a number by a given exponent of base 10 (10exponent), and formats it into a string representation of the number."""
    try:
        assert decimals > 0
        assert value >= 0
        result = value / 10 ** decimals
        return json.dumps({"result": result})
    except Exception as e:
        print(e)
        return json.dumps({"error": True})