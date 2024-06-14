import abc

import pandas as pd


class SixMwtSummary(abc.ABC):

    @abc.abstractmethod
    def get_full_df(self) -> pd.DataFrame:
        raise NotImplementedError


class LocalSixMwtSummary(SixMwtSummary):

    def __init__(self, path: str):
        self.path = path

    def get_full_df(self) -> pd.DataFrame:
        df = pd.read_parquet(self.path)
        df.BiologicalSex = df.BiologicalSex.apply(lambda x: 1 if x == "Male" else 0)
        return df
