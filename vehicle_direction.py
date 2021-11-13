import numpy as np
from scipy.spatial import distance

def find_angle_distance(points):
    d = calculate_covered_distance(points[-20:])
    if(d > 30):
        points = points[-40:]
        size = len(points)//4
        points = points[::size]
        p1,p2,p3,p4 = points[-4:]
        
        if(calculate_covered_distance([p2,p4]) > 20):
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p4) - np.array(p3)
            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            angle = np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))
            # print(angle)
            if(0<=angle<=23):
                return "straight"
            
            elif(angle>23 and angle<90):
                # diff = v2[0] - v1[0]
                A,B,C = p1,p2,p4
                """
                Assuming the points are (Ax,Ay) (Bx,By) and (Cx,Cy), you need to compute:
                (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
                This will equal zero if the point C is on the line formed by points A and B, and will have a different sign depending on the side. Which side this is depends on the orientation of your (x,y) coordinates, but you can plug test values for A,B and C into this formula to determine whether negative values are to the left or to the right                        
                """
                diff = (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])

                if(diff > 0 ):
                    return "right"
                elif(diff<0):
                    return "left"
                else:
                    return "straight"
            else:
                return "None"    
        else:
            return "None"

    
    else:
        return "stopped"

    
def calculate_covered_distance(points):
    d = 0        
    for i in range(len(points)-1):
        d+=distance.euclidean(points[i],points[i+1])
    return d

