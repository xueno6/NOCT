from BasisFunctions import BasisFunction
import pandas as pd
class ValueGenerator:
    def __init__(self, Dict_of_BF_code):
        self.__Dict_of_BF_code=Dict_of_BF_code
        self.BFs=[]
        for BF_code in self.__Dict_of_BF_code.keys():
            self.BFs.append(BasisFunction(BF_code))
    def __call__(self, dataRow):
        result=[]
        for BF, BF_code in zip(self.BFs, self.__Dict_of_BF_code.keys()):
            lst, success = BF(self.select_elements(dataRow, self.__Dict_of_BF_code[BF_code]))
            if success: 
                result.extend(lst)
            else:
                raise ValueError("Recheck the input of basis function"+BF.expression)
        return result
    def select_elements(self, data, indices):
        # Check if the input data is a list
        if isinstance(data, list):
            try:
                return [data[i] for i in indices]
            except:
                raise ValueError("index may be out of bound")
        # Check if the input data is a DataFrame row
        elif isinstance(data, pd.Series):
            try:
                return [data.iloc[i] for i in indices]
            except:
                raise ValueError("index may be out of bound")
        else:
            raise ValueError("Input should be either a list or a DataFrame row.")


from BasisFunctionsNames import BasisFunctionName
class ExpressionGenerator:
    def __init__(self, Dict_of_BF_code):
        self.__Dict_of_BF_code=Dict_of_BF_code
        self.BFNs=[]
        for BF_code in self.__Dict_of_BF_code.keys():
            self.BFNs.append(BasisFunctionName(BF_code))
    def __call__(self, colnames):
        result=[]
        for BFN, BF_code in zip(self.BFNs, self.__Dict_of_BF_code.keys()):
            lst, success = BFN(self.select_elements(colnames, self.__Dict_of_BF_code[BF_code]))
            if success: 
                result.extend(lst)
            else:
                raise ValueError("Recheck the input of basis function"+BFN.expression)
        return result
    def select_elements(self, data, indices):
        # Check if the input data is a list
        if isinstance(data, list):
            try:
                return [data[i] for i in indices]
            except:
                raise ValueError("index may be out of bound")
        # Check if the input data is a DataFrame row
        elif isinstance(data, pd.Series):
            try:
                return [data.iloc[i] for i in indices]
            except:
                raise ValueError("index may be out of bound")
        else:
            raise ValueError("Input should be either a list or a DataFrame row.")