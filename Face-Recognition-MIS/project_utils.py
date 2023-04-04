import pyodbc
from datetime import datetime
import pandas as pd
import os
import sys
import time


class AttendanceManagement:
    def __init__(self):
        self.current_date = datetime.now().strftime("%D")
        self.present_persons = get_present_persons(self.current_date)
        self.connection = connect_to_db()
        self.cursor = self.connection.cursor()
        self.today = datetime.now().strftime("%D")

    def data_write(self, detected_names):
        if datetime.now().strftime("%H:%M") == '01:30':
        # if datetime.now().strftime("%D") != self.today:
            # print(datetime.now().strftime("%H:%M:%S"), self.today)
            self.restart_DB()
            # self.today = datetime.now().strftime("%D")
        for name in detected_names:
            if name != 'unknown':
                if name not in self.present_persons:
                    self.present_persons.append(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    self.insert_db(self.current_date, name, 100, current_time)

    def insert_db(self, current_date, name, staffID, current_time):
        sql = f"INSERT INTO MISAttendance (CurrentDate, PersonName, StaffID, ArrivalTime) " \
              f"VALUES ('{current_date}','{name}','{staffID}','{current_time}')"

        try:
            self.cursor.execute(sql)
            self.connection.commit()
            print(f'Attendance Recorded for {name} at {current_time}')
        except Exception as ex:
            print(ex)

    def restart_DB(self):
        self.current_date = datetime.now().strftime("%D")
        restart_time = datetime.now().strftime("%H:%M:%S")
        self.present_persons = get_present_persons(self.current_date)
        print(f'Face Dataset Restart at : {restart_time}')
        time.sleep(35)


def write_to_db(records):
    coor_details_df = pd.DataFrame().from_records(records,
                                                  columns=['CurrentDate', 'PersonName', 'StaffID', 'ArrivalTime'])
    coor_details_df.to_csv('Test_save_DB.csv', index=False)
    os.system("bcp MISAttendance in Temp_attendance.csv -S 192.168.100.161,1433 -U sa -P dataport -d Test_DB -F 2 -c -t  ','")
    os.remove('Temp_attendance.csv')


def connect_to_db():
    odbc_driver = '{ODBC Driver 17 for SQL Server}'
    native_driver = '{SQL Server Native Client 11.0}'
    driver = native_driver
    connection = None
    if sys.platform == 'linux':
        driver = odbc_driver
    else:
        driver = native_driver
    connection = pyodbc.connect(
        f"""Driver={driver};
        Server=192.168.100.161,1433;
        Trusted_Connection=no;
        Database=Test_DB;
        UID=sa;
        PWD=dataport;"""
    )
    return connection

def get_present_persons(current_date):
    connection = connect_to_db()
    cursor = connection.cursor()
    SQL_COMMAND = f"select PersonName from MISAttendance cdt where (CurrentDate = '{current_date}') "
    cursor.execute(SQL_COMMAND)
    data = cursor.fetchall()
    persons = []
    for customer in data:
        for item in customer:
            persons.append(item)
    return persons