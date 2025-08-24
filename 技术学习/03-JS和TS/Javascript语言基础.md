本质是一种脚本语言。
# JS变量与变量类型

## 变量声明

- `const`：声明常量
- `var`和 `let`：声明变量
	- 推荐全部使用`let`防止混乱。
	> 查看函数部分讲解

```js
var a = 0; a = 1; // OK 
let b = "Test string";
b = "New string"; // OK
const c = 0; 
c = 2; // Error!
```
JS是弱类型语言，在变量赋值时可以改变类型。
```js
let weakType = 0; 
weakType = "You are a string now!"; // OK
```

JavaScript 语言支持七种基本类型，即数字、大整数、字符串、布尔值、`symbol` 类型、`undefined` 类型和 `null` 类型。
本文在默认习惯于C++等语言，仅讲解JS中的特点。


## `typeof`获取类型

```js
let testNum = 0; 
typeof testNum; // "num
```

## 布尔、数字和字符串

### 布尔

== 和 ！= 在JS中也可是用于判断是否等值，但是在比较时候会强制类型转换。所以更为推荐使用=== 和 !=== 进行对比：
```js
1 == true; // true 
1 === true; // false
```

### 数值
不区别int和float，统一使用浮点数表示。

仅当在 `Number.MAX_SAFE_INTEGER` 和 `Number.MIN_SAFE_INTEGER` 之间的整数运算是安全的，否则将会是使用双浮点数的近似值。所以除法没有类似 C/C++ 的向下取整的性质。

#### NaN
`NaN`不合法的运算结果。参与运算也只会得到`NaN`。但是在比较运算符中：

```js
NaN < 1; // false 
NaN > 1; // false 
NaN === NaN; // false 
NaN !== NaN; // true
```

只能使用`isNaN()`判断是否是`NaN`，这个函数**首先会将参数强制转化为数字类型**，并且只在转化结果为 `NaN` 的时候返回 `true`：
```js
isNaN(NaN); // true 
isNaN(3); // false 
isNaN("2.4"); // false, "2.4" can be converted to 2.4 
isNaN("No"); // true, "No" can never be converted to legal number 
 isNaN(true); // false, true can be converted to 1
```

#### Infinity
`Infinity` 代表无穷大，如果运算结果超出了 JavaScript 能处理的范围，则会得到 `Infinity`。`Infinity` 所参与的算术运算也一般符合数学直觉，如果涉及到不定式（零乘以无穷大、无穷大减无穷大等）则会得到 `NaN`：
```js
1 / 0; // Infinity
-1 / 0; // -Infinity
1e100000; // Infinity
0 / 0; // NaN

Infinity + 1; // Infinity
Infinity - 1; // Infinity
Infinity * 2; // Infinity
Infinity / 3; // Infinity

Infinity * 0; // NaN
Infinity - Infinity; // NaN

Infinity === Infinity; // true
```

### string
JavaScript 可以任意使用单双引号来表示字符串。JavaScript 中的 `string` 是原始值，也即 `string` 是不可改变的，这与 Python 类似，对 `string` 的任何操作会返回新的 `string` 值，而不是对旧的值做了部分修改。

JavaScript 的字符串支持使用加法运算符拼接，同时也支持相当多的常用函数。

可以查看这个 [JS Doc](https://devdocs.io/javascript-string/)

这里需要介绍的是**模板字符串**，这种字符串不使用单引号或双引号包围，而是使用**反引号**包围，内部可以使用 `${}` 块包围代码块，JavaScript 会计算出代码块的结果并将其转化为字符串嵌入模板之中。这个语法的好处在于不需要手写很多 `+` 来手动拼接字符串：

```js
let i = 1;
`The val of i + 1 is ${i + 1}.`; // "The val of i + 1 is 2."
```

最好不要在同一段代码中混用单双引号，也不要用反引号写非模板字符串。

JavaScript **允许任意变量和字符串相加**。而最常用的是字符串在加号左侧，其他变量在加号右侧的形式，这种运算的逻辑是将其他变量转化为字符串后进行字符串拼接。这就诞生了一个 trick，**即用一个空字符串加一个变量，就可以方便地将这个变量转化为字符串**：
```JS
"4" + 3; // "43"

"" + 3; // "3"
"" + true; // "true"
```

而将字符串转换为数字则可以使用 `parseInt` 和 `parseFloat` 函数，这里讲解 `parseInt` 函数。这个函数接受两个参数，第一个是需要转换的字符串，第二个是转换的进制数。不传入这个参数的时候默认根据字符串格式确定，如果以 `0x` 开头，则按照十六进制转换，其他则按照十进制转换.

`parseInt` 函数在转换失败的时候会返回 `NaN`。但要注意的是，这个函数的逻辑是逐个读取字符并实时转换，遇到不能转换的字符的时候返回已经转换好的结果而非 `NaN`.

#### bigint
`bigint` 类型用于存储和计算超过 `number` 类型限制的大数。

#### symbol
`symbol` 是 JavaScript 中非常独特的值，`symbol` 值只能通过调用 `Symbol()` 构造，传入的参数除用于调试外无其他意义，**该函数每次调用都会返回不同的 `symbol` 值**，故 `symbol` 类型值的唯一作用就是作为**独一无二的标识符**。

 



# JS的控制语句

# JS的函数

# JS的面向对象

# JS的异步

# JS的面向对象进阶

# 浏览器中的JS


# 参考

1. [SATA skill docs](https://docs.net9.org/languages/javascript/): 清华大学暑期课程分享文档
2. 