#Task2：数理统计与描述性分析（2天）

##理论部分(已理解)
统计量与抽样；常用统计量；
数据集中与离散趋势的度量；
分布特征，偏度与峰度；

##练习部分(已完成)
做理论知识点的笔记；
python实现数据各维度的描述性分析；

#拓展部分
## Python中的scipy.stats.mode函数
    Python中的scipy.stats.mode函数，寻找出现次数最多的成员
    from scipy.stats import mode
    
    list = ['a', 'a', 'a', 'b', 'b', 'b', 'a','b']
    print("# Print mode(list):", mode(list))
    print("# list中最常见的成员为：{}，出现了{}次。".format(mode(list)[0][0], mode(list)[1][0]))
    
    Print mode(list): ModeResult(mode=array(['a'], dtype='<U1'), count=array([4]))
    结果为：list中最常见的成员为：a，出现了4次；
    
    """
    def mode(a, axis=0, nan_policy='propagate'):
    函数作用：返回传入数组/矩阵中最常出现的成员以及出现的次数。
    
    如果多个成员出现次数一样多，返回值小的那个。 #比如上面a,b结果均为4次，最后输出较小的值a;
    
    """
    a = np.array([[2, 2, 2, 1],
                  [1, 2, 2, 2],
                  [1, 1, 3, 3]])
    print("# Print mode(a):", mode(a))
    print("# Print mode(a.transpose()):", mode(a.transpose()))
    print("# a的每一列中最常见的成员为：{}，分别出现了{}次。".format(mode(a)[0][0], mode(a)[1][0]))
    print("# a的第一列中最常见的成员为：{}，出现了{}次。".format(mode(a)[0][0][0], mode(a)[1][0][0]))
    print("# a的每一行中最常见的成员为：{}，分别出现了{}次。".format(mode(a.transpose())[0][0], mode(a.transpose())[1][0]))
    print("# a中最常见的成员为：{}，出现了{}次。".format(mode(a.reshape(-1))[0][0], mode(a.reshape(-1))[1][0]))
    
    list = ['a', 'a', 'a', 'b', 'b', 'b', 'a']
    print("# Print mode(list):", mode(list))
    print("# list中最常见的成员为：{}，出现了{}次。".format(mode(list)[0][0], mode(list)[1][0]))
    
    
    a的每一列中最常见的成员为：[1 2 2 1]，分别出现了[2 2 2 1]次。
    a的第一列中最常见的成员为：1，出现了2次。
    a的每一行中最常见的成员为：[2 2 1]，分别出现了[3 3 2]次。
    a中最常见的成员为：2，出现了6次。


## python中plt.hist参数详解

    x : (n,) array or sequence of (n,) arrays
    
    这个参数是指定每个bin(箱子)分布的数据,对应x轴
    
    bins : integer or array_like, optional
    
    这个参数指定bin(箱子)的个数,也就是总共有几条条状图
    
    normed : boolean, optional
    
    If True, the first element of the return tuple will be the counts normalized to form a probability density, i.e.,n/(len(x)`dbin)
    
    这个参数指定密度,也就是每个条状图的占比例比,默认为1
    
    color : color or array_like of colors or None, optional
    
    这个指定条状图的颜色
    
 #git
    (base) C:\Users\ccs\PycharmProjects>git commit -m 'git'
[master a13b72f8] 'git'
     1 file changed, 3 insertions(+)
     create mode 100644 PycharmProjects/.gitattributes
    
    (base) C:\Users\ccs\PycharmProjects>git push
    Enumerating objects: 17364, done.
    Counting objects: 100% (17364/17364), done.
    Delta compression using up to 8 threads
    Compressing objects: 100% (12277/12277), done.
    Writing objects: 100% (17363/17363), 343.41 MiB | 200.14 MiB/s, done.
    Total 17363 (delta 3943), reused 17240 (delta 3933)
    remote: Resolving deltas: 100% (3943/3943), completed with 1 local object.
    remote: warning: File PycharmProjects/t3_prod/t3_prod.zip is 64.88 MB; this is larger than GitHub's recommended maximum file si
    ze of 50.00 MB
    remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
    remote: error: Trace: a65596ce9f7b841d6fd3818a8d8e990a
    remote: error: See http://git.io/iEPt8g for more information.
    remote: error: File PycharmProjects/xx.zip is 132.35 MB; this exceeds GitHub's file size limit of 100.00 MB
    remote: error: File PycharmProjects/xx.zip is 118.46 MB; this exceeds GitHub's file size limit of 100.00 MB
    To https://github.com/ccs258/probability_theory.git
     ! [remote rejected]   master -> master (pre-receive hook declined)
    error: failed to push some refs to 'https://github.com/ccs258/probability_theory.git'
       
      #原因是git status发现把C盘其他文件也加入进来了；在项目目录下执行git checkout . && git clean -xdf
      ，再在上一级目录执行(venv) C:\Users\ccs>git checkout . && git clean -xdf
      再git status查看



#参考
    Python中的scipy.stats.mode函数使用：https: // blog.csdn.net / kane7csdn / article / details / 84795405
    python中plt.hist参数详解：https://www.cnblogs.com/python-life/articles/6084059.html
