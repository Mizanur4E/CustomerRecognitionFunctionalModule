import cv2
import sys
import pyodbc
import sklearn


def _connect_to_DB():
    odbc_driver = '{ODBC Driver 17 for SQL Server}'
    native_driver = '{SQL Server Native Client 11.0}'

    driver = native_driver
    conn = None
    if sys.platform == 'linux':
        driver = odbc_driver
    else:
        driver = native_driver
    conn = pyodbc.connect(
        f"""Driver={driver};
        Server=192.168.100.161;
        Trusted_Connection=no;
        Database=ShwapnoCustomerFaceEmbeddings;
        UID=sa;
        PWD=dataport;"""
    )
    return conn




class CustomerRecognizer():

    def __init__(self,cam_ip):
        print("initializing the recognizer...!")
        self.cam_ip = cam_ip
        print(self.cam_ip)
        while True:
            cus_id = self.faceDetector()
            #use the cus_id to assign the invoice

    def faceDetector(self):

        '''Detect face and generate body and face embedding vector'''

        CFD = False
        while True:

            # write code to check whether expected image frame is presented
            if CFD:
                break
            else:
                pass

        self.cus_BE = None      #generate customer body embedding and face embedding
        self.cus_FE = None
        print('Done Detection!')
        id = self.faceMatching()
        return id

    def faceMatching(self):

        '''Search DB against face embedding and return cusomer id'''

        #write some code for searching for the face embedding

        self.connection = _connect_to_DB()
        self.connection.autocommit = True
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM mytable")
        rows = cursor.fetchall()
        for row in rows:
            a = row.embedding.split(' ')
            #print(a)
            #measure cosine similarity between self.cus_FE and a
            sim_ind = sklearn.metrics.pairwise.cosine_similarity(a,self.cus_FE)
            if sim_ind > 0.75:
                return  row.id
        #generate id
        gen_cus_id = row.id + 1
        gen_no = row.no + 1
        new_emb = str(self.cus_FE)
        #write query to save the embedding with the gen id in the DB
        query2 = "INSERT INTO mytable(no,id,name,embedding) VALUES ((?),(?),'unkwn',(?))"
        cursor.execute(query2,gen_no,gen_cus_id,new_emb)
        return gen_cus_id

    def camera_test(self):
        cap = cv2.VideoCapture(self.cam_ip)
        while True:
            ret, frame = cap.read()
            print(ret)
            if ret:
                frame = cv2.resize(frame, (1920, 1080))
                # frame = cv2.resize(frame,(800,600))

                cv2.imshow('F', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # cv2.imwrite('cam2.jpg',frame)
                    break

        cv2.destroyAllWindows()


if __name__ == '__main__':

    ip = 'rtsp://admin:AiBi@8899@192.168.102.80:554'

    cus_recognizer = CustomerRecognizer(ip)
    #cus_recognizer.camera_test()