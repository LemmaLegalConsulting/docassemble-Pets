from docassemble.base.util import (
    as_datetime,
    DADateTime,
    DADict,
    DAList,
    DAObject,
    path_and_mimetype,
    today,
)
import pandas as pd
import os
from typing import List, Union, Dict, Iterable, Optional
from collections import OrderedDict

__all__ = [
    "DataLoader",
    "Pet",
    "PetsDict",
    "unique_values",
    "rows_with_label",
]

class BaseDataLoader(DAObject):
    """
    Object to hold some methods surrounding loading/filtering data.
    Built around Pandas dataframe.
    """

    def filter_items(
        self,
        display_column: Union[List[str], str] = "name",
        allowed_types: list = None,
        filter_column=None,
    ) -> Iterable:
        """
        Return a subset of rows, with only the specified column and index.
        This is useful for showing a drop-down menu of choices.
        """
        return self.filter(
            display_column=display_column,
            allowed_types=allowed_types,
            filter_column=filter_column,
        ).items()

    def filter(
        self,
        display_column: Union[List[str], str] = "name",
        allowed_types: list = None,
        filter_column=None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Return the raw dataframe filtered to the specified column(s) and matching the specified pet(s).
        """
        df = self._load_data()
        if allowed_types and filter_column:
            # Return only the names for matching values in the specified column
            return df[df[filter_column].isin(allowed_types)][display_column]
        else:
            return df[display_column]

    def load_row(self, index: Union[int, str]) -> Union[pd.Series, pd.DataFrame]:
        """
        Retrieve all of the data in a single row of the DataFrame
        """
        df = self._load_data()
        try:
            row = df.loc[index]
        except:
            return pd.DataFrame()
        return row

    def load_rows(self, loci: List[Union[int, str]]) -> pd.DataFrame:
        """
        Retrieve a slice of the dataframe, using the provided loci (indexes) as the basis for
        retrieval.
        """
        df = self._load_data()
        try:
            rows = df.loc[loci]
            return rows
        except:
            return pd.DataFrame()

    def get_rows(self, allowed_types: list = None, filter_column=None) -> pd.DataFrame:
        """
        Return a subset of rows, but with all columns.
        """
        df = self._load_data()
        if allowed_types and filter_column:
            # Return only the names for matching values in the specified column
            return df[df[filter_column].isin(allowed_types)]
            # return df[df[search_column].isin([category])]
        else:
            return df

    def _load_data(self) -> pd.DataFrame:
        """
        Return dataframe of the whole file's contents.
        """
        if "/" in self.filename:
            to_load = path_and_mimetype(self.filename)[0]
        else:
            to_load = path_and_mimetype(os.path.join("data/sources", self.filename))[0]

        if self.filename.lower().endswith(".xlsx"):
            df = pd.read_excel(to_load)
        elif self.filename.lower().endswith(".csv"):
            df = pd.read_csv(to_load)
        elif self.filename.lower().endswith(".json"):
            # TODO: we may need to normalize a JSON file
            df = pd.read_json(to_load)
        else:
            raise Exception(
                "The datafile must be a CSV, XLSX, or JSON file. Unknown file type: "
                + to_load
            )
        return df
      
class DataLoader(BaseDataLoader):
    def _load_data(self) -> pd.DataFrame:
        df = super()._load_data()
        df.set_index(
            "ID", inplace=True
        )  # Our XLSX file has a column 'ID' with a unique identifier for each row
        return df
      
class Pet(DAObject):
    pass
      
class PetsDict(DADict):
    def init(self, *pargs, **kwargs):
        super().init(*pargs, **kwargs)
        self.object_type = Pet
        self.complete_attribute = "complete"

    def as_merged_list(self):
        """Merge pet details with original DF row"""
        results = pd.concat([self[c].df for c in self])

    def as_list(self, language:str="en"):
        flattened = DAList(auto_gather=False, gathered=True)
        for species in self.elements:
            for index, row in self[species].df.iterrows():
                pet = self[species].details[index]
                pet.index = index
                pet.blurb = row.get("Blurb")
                pet.breed = row.get("Breed")
                pet.label = row.get("Label")
                pet.row = row
                flattened.append(pet)
        return flattened


def filter_df(
    dataloader: DataLoader,
    filter_column: Optional[str]=None,
    filter_value: Optional[str]=None,
    filter_column2: Optional[str]=None,
    filter_value2: Optional[str]=None,
) -> pd.DataFrame:
    df = dataloader._load_data()
    ## TODO could make this work more generalized by passing a dictionary
    ## https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
    if filter_column and filter_column2:
        return df[(df[filter_column] == filter_value) & (df[filter_column2] == filter_value2)]
    elif filter_column and filter_value:
        return df[df[filter_column] == filter_value]
    return df

def unique_values(
    dataloader: DataLoader,
    search_column: str,
    #limits: Optional[Dict[str,str]]=None,
    filter_column: Optional[str]=None,
    filter_value: Optional[str]=None,
    filter_column2: Optional[str]=None,
    filter_value2: Optional[str]=None,
):
    return list(
        filter_df(
            dataloader=dataloader,
            filter_column=filter_column,
            filter_value=filter_value,
            filter_column2=filter_column2,
            filter_value2=filter_value2,
        )[search_column].unique()
    )

def rows_with_label(
    dataloader: DataLoader,
    display_column: str,
    filter_column: Optional[str]=None,
    filter_value: Optional[str]=None,
    filter_column2: Optional[str]=None,
    filter_value2: Optional[str]=None,
) -> List[Dict[int,str]]:
    filtered_rows = filter_df(
        dataloader=dataloader,
        filter_column=filter_column,
        filter_value=filter_value,
        filter_column2=filter_column2,
        filter_value2=filter_value2,
    )
    return [
        {row[0]: row[1][display_column]} # Row 0 is the index
        for row
        in filtered_rows.iterrows()
    ]