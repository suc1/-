# [Python 全部 71 个内置函数](https://zhuanlan.zhihu.com/p/666496949)

## 输入输出
```
print()
input()		# type = str
```

## 基本数据类型
```
b = int('2')
b = float('2.0')
b = bool(2)
data1 = 1 + 2j						# complex()：复数类型
string = str(123456)

list2 = list(range(1, 10))
list2 = [ i for i in range(10)]

tuple2 = tuple("hello")				# ('h', 'e', 'l', 'l', 'o')

set1 = set([1, 2, 3, 4, 5])			# {1, 2, 3, 4, 5}
set2 = set(('python', 'xiaolong'))	# {'python', 'xiaolong'}	# 只拆解一层
set3 = set('suc')					# {'u', 's', 'c'}

keys = ['苹果', '梨', '香蕉']
values = [10, 20, 30]
dict2 = dict(zip(keys, values))		# {'苹果': 10, '梨': 20, '香蕉': 30}
```

l = filter(lambda i: i%2, list1)	# filter object
list(l)
l = map(lambda i: i+1, list1)	# 对可迭代对象中的元素进行映射，分别去执行自定义函数
## 数学运算
```
a = abs(-11)
a = divmod(5, 3)					# 商和余数(1, 2)
a = round(10.2453, 2)				# 10.25, 四舍五入
a = pow(10, 2)

list1 = [1, 2, 3, 4, 5]
a = sum(list1)
a = min(list1)
a = max(list1)
```

## 进制转换
```
a = bin(6)							# str
a = oct(8)
a = hex(8)
```

## 序列集合操作
```
list1 = [1, 2, 3, 4, 5]
list(reversed(list1))				# 返回一个反向的迭代器
list1[1:3]

b = bytes('@程序员小龙', encoding='utf-8')	# bytes

print(ord('a'))   					# 97	# 编码
print(ord('小'))  					# 31243

for i in range(49, 58):
    print(chr(i), end=' ')			# str
	
print(ascii('Hello,@程序员小龙\n'))	# str
s = 'Hello,\t@程序员\n小龙\n'
print(repr(s)) 						# 'Hello,\t@程序员\n小龙\n'		# 保留转义字符

len(list1)

sorted(list1, reverse=True)			# 排序
# 自定义规则
list2 = ['one', 'two', 'three', 'four', 'five', 'six']
def func(s):
    return len(s)
print(sorted(list2, key=func))

score_dict = {
    '张三': 33,
    '李四': 36}						# score_dict.keys(), score_dict.values(), score_dict.items()
score_dict_sorted = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
for index, (name, score) in enumerate(score_dict_sorted, start=1):
    print(f'第{index}名：{name}\t成绩：{score}')
	
all()
any()
zip()								# 组成元组
l = filter(lambda i: i%2, list1)
l = map(lambda i: i+1, list1)
```

## 文件操作
```
f = open("name.txt", 'r', encoding='utf-8')
f.read()
f.close()
```

## 迭代器相关
```
range()
next()
iter()：获取迭代器					# next(it)
```

## 作用域相关（用于查错）
```
locals()							# 返回当前作用域中的变量名字
globals()							# 返回全局作用域中的变量名字
```

## 其他
```
help(list)
dir()								# 查看对象的内置属性
callable()							# 用于检查一个对象是否可以调用. 类是可以被调用的; 实例是不可以被调用的，除非类中声明了__call__方法
hash()
__import__()						# 用于动态加载类和函数
```

## [python内置函数大全](https://zhuanlan.zhihu.com/p/61977192)
```
isinstance(basestring)				# str, unicode
unichr()
frozenset()
xrange()							# 需要时计算，= range()
cmp(x, y)
classmethod()						# 类方法即可被类调用，也可以被实例调用
compile(source, filename, mode[, flags[, dont_inherit]])	# 编译为代码或者AST对象。代码对象能够通过exec语句来执行或者eval()进行求值single
delattr(object, name)
eval(expression [, globals [, locals]])
execfile(filename [, globals [, locals]])

```