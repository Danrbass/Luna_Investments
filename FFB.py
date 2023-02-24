# pylint: disable=import-error
import pandas as pd
import pandas_datareader.data as reader
import statsmodels.api as sm
import FactorsBrazil as FB

class Acoes():
    def __init__(self, alpha, omega):
        self.alpha = alpha
        self.omega = omega

    def indice(self, indices):
        return reader.get_data_yahoo(indices, start=self.alpha, end=self.omega)['Adj Close'].pct_change()

    def frequencia_indices(self, indices, frequencia):
        return Acoes(self.alpha, self.omega).indice(indices).resample(frequencia).agg(lambda x: (x + 1).prod() - 1)[1:]

    def indices(self, indice, frequencia):
        cotacao_indices = Acoes(self.alpha, self.omega).frequencia_indices(indice, frequencia)
        return pd.DataFrame(cotacao_indices).rename(columns={'Adj Close': indice})

    def cotacao(self, tickers):
        return reader.get_data_yahoo([t + '.SA' for t in tickers], start=self.alpha, end=self.omega)['Adj Close'].pct_change()

    def frequencia_acoes(self, tickers, frequencia):
        return Acoes(self.alpha, self.omega).cotacao(tickers).resample(frequencia).agg(lambda x: (x + 1).prod() - 1)[1:]

    def cotacao_acoes(self, tickers, frequencia):
        cotacao_acoes = Acoes(self.alpha, self.omega).frequencia_acoes(tickers, frequencia)
        return pd.DataFrame(cotacao_acoes).rename(columns={i + '.SA': i.replace('.SA', '') for i in tickers})


class DFTreatment(Acoes):
    def __init__(self, tickers, rf, alpha, omega):
        super().__init__(alpha, omega)

        self.tickers = tickers
        self.rf = rf

    def rm_minus_rf(self, on: str, dcolumns: str):
        du = pd.merge(self.tickers, self.rf, on=on)
        ds = [du[i] - du['Risk_free'] for i in du.columns]
        dc = pd.concat(ds, axis=1, ignore_index=True, verify_integrity=True)
        return dc.rename(columns=dict(zip(dc.columns, du.columns))).drop(columns=dcolumns)

    def mergedffull(self, on: str, dcolumns: str, alpha, omega):
        df_full = DFTreatment(self.tickers, self.rf, alpha, omega).rm_minus_rf(on, dcolumns)
        ff5f = FB.Factors(alpha, omega).five_factors(dcolumns)

        return pd.merge(df_full, ff5f, on=on)


class Result(DFTreatment):
    def __init__(self, tickers, rf, alpha, omega):
        super().__init__(tickers, rf, alpha, omega)

        self.list_factors = ['Rm_minus_Rf', 'SMB', 'HML', 'IML', 'WML']

    def y(self, on: str, dcolumns: str, alpha, omega):
        df_full = DFTreatment(self.tickers, self.rf, alpha, omega).mergedffull(on, dcolumns, alpha, omega)
        return df_full.drop(columns=self.list_factors)

    def X(self, on: str, dcolumns: str, alpha, omega):
        df_full = DFTreatment(self.tickers, self.rf, alpha, omega).mergedffull(on, dcolumns, alpha, omega)
        return df_full[self.list_factors]

    def params(self, on: str, dcolumns: str, alpha, omega):
        y = Result(self.tickers, self.rf, alpha, omega).y(on, dcolumns, alpha, omega)
        X = Result(self.tickers, self.rf, alpha, omega).X(on, dcolumns, alpha, omega)
        return [sm.OLS(y[i], sm.add_constant(X)).fit().params for i in y.columns]

    def ff_coef_factors(self, on: str, dcolumns: str, alpha, omega):
        y = Result(self.tickers, self.rf, alpha, omega).y(on, dcolumns, alpha, omega)
        dr = Result(self.tickers, self.rf, alpha, omega).params(on, dcolumns, alpha, omega)
        dc = pd.concat(dr, axis=1, verify_integrity=True, ignore_index=False)
        return dc.rename(columns=dict(zip(dc.columns, y.columns))).T


class Stocksfactors(Result):
    def __init__(self, factors, tickers, rf, alpha, omega, on, dcolumns):
        super().__init__(tickers, rf, alpha, omega)

        self.factors = factors

        self.df = Result(self.tickers, self.rf, alpha, omega).ff_coef_factors(on, dcolumns, alpha, omega)

    def nstocks_smallest(self, n: int):
        return self.df.nsmallest(n, self.factors, keep='first').index

    def nstocks_largest(self, n: int):
        return self.df.nlargest(n, self.factors, keep='first').index
