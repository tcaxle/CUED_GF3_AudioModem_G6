"""
Module for demapping constellations
"""

class Constellation:
    """
    A class for holding constellation objects
    """
    def __init__(self, symbol_value_list):
        """
        Takes symbol_value_list -
            [(complex_symbol, (binary_data_tuple)), ...]
        """
        # Check all symbols and values are unique
        # And check all data of correct type
        symbol_list = []
        value_list = []
        for symbol, value in symbol_value_list:
            # Unique check
            if symbol in symbol_list or value in value_list:
                raise Exception("All symbols and values in a constellation must be unique")
            symbol_list.append(symbol)
            value_list.append(value)
            # Data type check
            if type(symbol) != complex or type(value) != tuple:
                raise Exception("All symbols must be complex numbers and all values must be tuples")

        # Define attributes
        self.symbol_value_list = symbol_value_list
        self.symbol_list = [symbol_value[0] for symbol_value in symbol_value_list]
        self.value_list = [symbol_value[1] for symbol_value in symbol_value_list]
        self.rank = len(symbol_value_list)

    def get_symbol_from_value(self, input_value):
        """
        Gets the symbol from the value
        """
        for symbol, value in self.symbol_value_list:
            if value == input_value:
                return symbol

    def get_value_from_symbol(self, input_symbol):
        """
        Gets the value from the symbol
        """
        for symbol, value in self.symbol_value_list:
            if symbol == input_symbol:
                return value

    def miminum_distance_demap(self, data):
        """
        Takes two arguments:
            constllation - a Constellation object
            data - a list of complex values which we wish
                to map to our constellation
        Returns two things:
            A dictionary of {symbol: distance}
        """
        output_dict = {}
        for datum in data:
            # Get distance from datum to all symbols in constellation
            distances = {abs(datum - symbol): symbol for symbol in self.symbol_list}
            minimum_distance = min(distances.keys())
            symbol = distances[minimum_distance]
            output_dict[symbol] = minimum_distance
        return output_dict
