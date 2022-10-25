import os
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

'''
迪杰斯特拉算法（Dijkstra）
https://blog.csdn.net/qq_40347399/article/details/119820276
'''

#北京 天津 郑州 济南 长沙 海南
# 0    1    2    3    4    5
 
#模拟从文件中读入图的各个路径
a = """
0 1 500
0 2 100
1 2 900
1 3 300
2 3 400
2 4 500
3 4 1300
3 5 1400
4 5 1500
"""
 
INF = float('inf')
 
#定义邻接矩阵 记录各城市之间的距离
weight = [[INF if j!=i else 0 for j in range(6)] for i in range(6)]
 
#解析数据
b = [[int(i) for i in i.split(' ')] for i in a.split('\n') if i != '']
 
for i in b:
    weight[i[0]][i[1]] = i[2]
    weight[i[1]][i[0]] = i[2]
 
def dijkstra(src, target):
    """
    src : 起点索引
    dist: 终点索引
    ret:  最短路径的长度
    """
    #未到的点
    u = [i for i in range(6)]
    #距离列表
    dist = weight[src][:]
    #把起点去掉
    u.remove(src)
    
    #用于记录最后更新结点
    last_update = [src if i != INF else -1 for i in dist]
 
    while u != []:
        
        idx = 0
        min_dist = INF
        
        #找最近的点
        for i in range(6):
            if i in u and dist[i] < min_dist:
                min_dist = dist[i]
                idx = i
 
        #从未到列表中去掉这个点
        u.remove(idx)
        
        #更新dist（借助这个点连接的路径更新dist）
        for j in range(6):
            if j in u and weight[idx][j] + min_dist < dist[j]:
                dist[j] = weight[idx][j] + min_dist
                
                #记录更新该结点的结点编号
                last_update[j] = idx
 
    #输出从起点到终点的路径结点
    tmp = target
    path = []
    while tmp != src:
        path.append(tmp)
        tmp = last_update[tmp]
    path.append(src)
    logging.info("->".join([str(i) for i in reversed(path)]))
 
    return dist[target]    
 
 
if __name__ == '__main__':
    src, dst = 0, 5
    dijkstra(src, dst)
    
    logging.info('over')
