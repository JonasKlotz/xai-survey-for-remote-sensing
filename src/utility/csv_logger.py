import os
from typing import List


class CSVLogger:
    def __init__(
        self,
        log_dir: str,
        filename: str = "metrics",
        append: bool = False,
        column_names: List[str] = None,
        separator: str = ";",
    ):
        """CSV Logger for logging metrics to a csv file.

        Parameters
        ----------
        log_dir : str
            directory where the log file should be stored
        filename : str
            name of the log file
        append : bool
            whether to append to the log file or overwrite it
        column_names : List[str]
            list of column names
        separator : str
            separator for the csv file
        """
        self.log_dir = log_dir
        self.filename = filename
        self.append = append
        self.column_names = column_names
        self.separator = separator

        self.log_file_path = None

        self._init_log_file()

    def _init_log_file(self):
        complete_filename = f"{self.filename}.csv"
        # create log directory if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file_path = os.path.join(self.log_dir, complete_filename)

        if not self.append and self.column_names is not None:
            self._write_column_names()

    def _write_column_names(self):
        with open(self.log_file_path, "w") as f:
            f.write(self._get_column_names_as_string())

    def _get_column_names_as_string(self):
        return self.separator.join(self.column_names) + "\n"

    def update(self, row: dict):
        """Append a row to the csv file

        Creates a new csv file if it does not exist yet.

        Parameters
        ----------
        row : dict
            row to append
        """
        # check if file already exists
        if not os.path.exists(self.log_file_path):
            self.column_names = list(row.keys())
            self._write_column_names()

        # check if column names are correct
        if self.column_names is None:
            raise ValueError("Column names not set. Please set them before appending.")

        # check if column names are correct
        if not set(self.column_names).issubset(set(row.keys())):
            raise ValueError("Column names do not match row keys.")

        # check if row values are correct
        if not len(self.column_names) == len(row.keys()):
            raise ValueError("Column names do not match row keys.")

        # append row to file
        with open(self.log_file_path, "a") as f:
            f.write(self._get_row_as_string(row))

    def _get_row_as_string(self, row: dict):
        return self.separator.join([str(row[key]) for key in self.column_names]) + "\n"
