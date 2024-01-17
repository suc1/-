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

## 进程和线程
1. 进程是资源分配的最小单位
2. 线程是程序执行的最小单元
3. 进程拥有自己的独立地址空间

## IPC 类型
1. 半双工管道
2. 命名管道
3. 消息队列
4. 信号
5. 信号量
6. 共享内存
7. 内存映射文件
8. 套接字


## [Linux内核中的各种锁](https://mp.weixin.qq.com/s?__biz=Mzg2MzQzMjc0Mw==&mid=2247494597&idx=1&sn=67dbd3275abea2f4e26fe8a63f0b5eee&chksm=ce7a128df90d9b9b59246da2c73c3d98cf57f3b8124367c66ca49a10b8b4e91d553f9804d9e4&scene=132&exptype=timeline_recommend_article_extendread_samebiz#wechat_redirect)
1. CPU: atomic原子变量，spinlock自旋锁. 针对多核处理器或多CPU处理器
2. 临界区: semaphore信号量，Mutex互斥锁，rw-lock读写锁，preempt抢占
3. cache: per-CPU, 解决各个CPU里L2 cache和内存间的数据不一致性
4. 内存: RCU, Memory Barrier
5. 原子变量: 对某个被多线程会访问
6. 互斥锁中，要是当前线程没拿到锁，就会出让CPU；而自旋锁中，要是当前线程没有拿到锁，当前线程在CPU上忙等待直到锁可用
7. 信号量本质是一个计数器, PV操作, 多线程通信: `sem_wait()`, `sem_post()`
8. 互斥锁只让一个线程进入, 等待超时`std::timed_mutex`
9. RCU=read copy update; 1）复制后更新；2）延迟回收内存; 读写可以并行
10. 内存屏障: 控制内存访问顺序; 1）编译器层面的, 2）CPU层面的
11. C++里的volatile关键字只能避免编译期的指令重排，对于多CPU的指令重排不起作用
12. 多CPU乱序访问内存：必须通过cache 的一致性协议来避免数据不一致
13. CPU级别的内存屏障: 1)通用 barrier; 2)写操作 barrier; 3)读操作 barrier
14. CAS即Compare and Swap, 基本思想是：1)读，修改，比较; 2) 使用内存屏障来保证操作的顺序和一致性

## [C++ 模板总结](https://mp.weixin.qq.com/s/175BheaXV4-vJ_T4_sdk2g)
1. ``