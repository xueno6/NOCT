# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:16:39 2023

@author: MSI-NB
"""
import re
import numpy as np
import itertools
class BasisFunctionName:
    def __init__(self, BF_code: str, expression='1'):
        '''
        Parameters
        ----------
        BF_code : str
            The BF_code is the type and detail of this basis fuction. it has 7 types
            BF_code structure: ABC-int, int, int, int...
            BF type:
            1. Polynomial, but with same power terms. 
               example PLY-3, 3, 3, each int corresponds the power terms
            2  Polynomial but with different power terms(must be different). 
               example PLZ-12, 3, 4
            3. e^((x/a)^b) as EXP-a, b. 
               example EXP-2, 3
            4. ln((x/a)^b) as LOG-a, b.
               example LOG-2, 3
            5. sin(x/a) as SIN-a, no more float needed.
            6. cos(x/a) as COS-a, no more float needed.
            7. self-designed USE-, must provide the expression at the same time.
        
        expression : str
            The default is '1'.
            No need for input if BF_code is 1-6 type
            if BF_code='USE-', expression must be a python function, the variables must start from 'X1' to 'XN'
            example '1/(1+np.exp(-X1))'

        Raises
        ------
        Exception
            The BF_code do not follow the description

        Returns
        -------
        None.

        '''
        self.__BF_code = BF_code
        self.expression = expression
        self.number_of_variables = 1
        if 'PLY' == self.__BF_code[0:3]:
            #homogeneous power terms
            power_terms, flag = self.string_to_tuple_and_check(self.__BF_code[4:])
            if False == flag:
                raise Exception('''Error inputs for Polynomial terms: may contain non-float type value, or numbers are not split by ', '.''')
            elif len(set(power_terms)) != 1:
                raise Exception('''Error inputs for Polynomial terms: may contain non-homogeneous power terms''')
            else:
                self.number_of_variables = len(power_terms)
                for index, power_term in enumerate(power_terms):
                    self.expression+='*X'+str(index+1)+'**'+str(power_term)
        elif 'PLZ' == self.__BF_code[0:3]:
            power_terms, flag = self.string_to_tuple_and_check(self.__BF_code[4:])
            if False == flag:
                raise Exception('''Error inputs for Polynomial terms: may contain non-float type value, or numbers are not split by ', '.''')
            elif len(set(power_terms)) == 1:
                raise Exception('''Error inputs for Polynomial terms: may contain homogeneous power terms''')
            else:
                self.number_of_variables = len(power_terms)
                for index, power_term in enumerate(power_terms):
                    self.expression+='*X'+str(index+1)+'**'+str(power_term)
        elif 'EXP' == self.__BF_code[0:3]:
            power_terms, flag = self.string_to_tuple_and_check(self.__BF_code[4:])
            if False == flag:
                raise Exception('Error inputs for Exponential terms: may contain non-float type value')
            elif len(power_terms)>2:
                raise Exception('Error inputs for Exponential terms: may extend two terms')
            else:
                self.expression='np.exp((X1/'+ str(power_terms[0])+ ')'+ '**'+str(power_terms[1])+ ')'
        elif 'LOG' == self.__BF_code[0:3]:
            power_terms, flag = self.string_to_tuple_and_check(self.__BF_code[4:])
            if False == flag:
                raise Exception('Error inputs for Logarithmic terms: may contain non-float type value')
            elif len(power_terms)>2:
                raise Exception('Error inputs for Logarithmic terms: may extend two terms')
            else:
                self.expression='np.log((X1/'+ str(power_terms[0])+ ')'+ '**'+str(power_terms[1])+ ')'
        elif 'SIN' == self.__BF_code[0:3]:
            power_terms, flag =self.string_to_tuple_and_check(self.__BF_code[4:])
            if False == flag:
                raise Exception('Error inputs for Trigonometric terms: may contain non-float type value')
            elif len(power_terms)>1:
                raise Exception('Error inputs for Trigonometric terms: may extend one terms')
            else:
                self.expression='np.sin(X1/'+ str(power_terms[0])+ ')'
        elif 'COS' == self.__BF_code[0:3]:
            power_terms, flag =self.string_to_tuple_and_check(self.__BF_code[4:])
            if False == flag:
                raise Exception('Error inputs for Trigonometric terms: may contain non-float type value')
            elif len(power_terms)>1:
                raise Exception('Error inputs for Trigonometric terms: may extend one terms')
            else:
                self.expression='np.cos(X1/'+ str(power_terms[0])+ ')'
        elif 'USE' == self.__BF_code[0:3]:
            # YOU MUST GIVE expression while use USE type
            self.expression = self.__BF_code[4:]
            matches = re.findall(r'X\d+', self.expression)
            self.number_of_variables = len(set(matches))
        
        
    def __call__(self, lst):
        '''
        Parameters
        ----------
        lst : lst or tuple of names(str)
            normally a tuple of numbers, for example bf(('x1','x2'))

        Returns
        -------
        result: list
        names from the input to this basis function.             
        bool
            True: sucess 
            False: No enough inputs

        '''
        names = tuple(lst)
        if len(names)< self.number_of_variables:
            return 'No enough inputs', False
        elif 'PLY' == self.__BF_code[0:3]:
            result = []
            for subnames in list(itertools.combinations(names, self.number_of_variables)):
                result.append(self.replace_variables(self.expression, subnames))
            return result, True
        elif 'PLZ' == self.__BF_code[0:3]:
            result = []
            for subnames in list(itertools.permutations(names, self.number_of_variables)):
                result.append(self.replace_variables(self.expression, subnames))
            return result, True
        elif 'USE' == self.__BF_code[0:3]:
            result = []
            for subnames in list(itertools.permutations(names, self.number_of_variables)):
                result.append(self.replace_variables(self.expression, subnames))
            return result, True
        else:
            return [self.replace_variables(self.expression, [name]) for name in names],True
    
    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def string_to_tuple_and_check(self, s: str):
        elements = s.split(', ')

        if all(self.is_float(element) for element in elements):
            return tuple(float(element) for element in elements), True
        else:
            return None, False
    def replace_variables(self, expression, new_vars):
        for i in range(self.number_of_variables):
            # Construct the old variable name
            old_var = f"X{i+1}"
            expression = expression.replace(old_var, new_vars[i])
        return expression

