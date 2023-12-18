# CPP
1. [C语言和C++的区别和联系](https://mp.weixin.qq.com/s/ubZJnZVk58nBXIOSNRqGFw)

## 多态
1. 静多态：函数重载，函数模板
2. 动多态（运行时的多态）：继承中的多态（虚函数）

## 数组的引用
```
int array[10] = {0}；	//array拿出来使用的话就是数组array的首元素地址。即是int *类型
int (*q)[10] = &array;	//数组指针
int (&q)[10] = array;	//数组引用
```

## 内存的申请与释放
1. malloc和free是函数，new和delete是运算符
2. malloc在分配内存前需要大小，new不需要
3. malloc不安全，需要手动类型转换，new不需要类型转换
4. new是先申请空间再调用构造函数（如果需要）
5. free只释放空间，delete先调用析构函数再释放空间（如果需要）
6. malloc失败返回0，new失败抛出bad_alloc异常
7. malloc开辟在堆区，new开辟在自由存储区域???
8. new可以调用malloc(),但malloc不能调用new

## 作用域
1. C语言中作用域：局部，全局
2. C++中则有：局部作用域，类作用域，名字空间作用域
