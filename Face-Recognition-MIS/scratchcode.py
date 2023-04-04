def IsInside(ref, face):

    if ref[0] < face[0] and ref[2] > face[2] and ref[1] < face[1] and ref[3] > face[3]:
        return True
    else:
        return False



def testZip():

    a = [1,2,3,4]
    b = [100,200,300,400]

    for x,y in zip(a,b):
        print(x,y)

testZip()

# ref = [0,0,4,4]
# face1 = [1,1,2,2]
# face2 = [1,1,5,4]
# face3 = [2,5,3,9]
# print(IsInside(ref,face1), IsInside(ref, face2), IsInside(ref, face3))