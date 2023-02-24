# pylint: disable=invalid-name
import pandas as pd


class Factors():

    def __init__(self, alpha, omega):
        self.omega = omega
        self.alpha = alpha
        self.list_factors = [
            'Market_Factor',
            'SMB_Factor',
            'HML_Factor',
            'IML_Factor', 
            'WML_factor', 
            'Risk_Free'
            ]

        self.data = [
            pd.read_excel(f'../Luna_Investimentos/factors/{data}.xls',
                          parse_dates={"Date": ['year', 'month', 'day']}).set_index('Date') for data in
            self.list_factors
        ]

    def fama_french(self):
        famafrench = pd.concat(self.data, axis=1, ignore_index=False)
        return famafrench.query(f' Date >= "{self.alpha}" and Date <= "{self.omega}" ')

    def five_factors(self, dcolumns):
        return Factors.fama_french(self).drop(columns=dcolumns)

    def market_risk(self):
        return Factors.fama_french(self)['Rm_minus_Rf']

    def smb(self):
        return Factors.fama_french(self)['SMB']

    def hml(self):
        return Factors.fama_french(self)['HML']

    def iml(self):
        return Factors.fama_french(self)['IML']

    def wml(self):
        return Factors.fama_french(self)['WML']

    def rf(self):
        return Factors.fama_french(self)['Risk_free']
