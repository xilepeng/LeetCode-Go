
[227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

[224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)


------

[227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

```go
func calculate(s string) int {
	stack, sign, num, res := []int{}, byte('+'), 0, 0
	for i := 0; i < len(s); i++ {
		isDigit := s[i] <= '9' && s[i] >= '0'
		if isDigit {
			num = num*10 + int(s[i]-'0')
		}
		if !isDigit && s[i] != ' ' || i == len(s)-1 {
			if sign == '+' {
				stack = append(stack, num)
			} else if sign == '-' {
				stack = append(stack, -num)
			} else if sign == '*' {
				stack[len(stack)-1] *= num
			} else if sign == '/' {
				stack[len(stack)-1] /= num
			}
			sign = s[i]
			num = 0
		}
	}
	for _, i := range stack {
		res += i
	}
	return res
}
```

- 加号：将数字压入栈；
- 减号：将数字的相反数压入栈；
- 乘除号：计算数字与栈顶元素，并将栈顶元素替换为计算结果。

```go
func calculate(s string) int {
	stack, preSign, num, res := []int{}, '+', 0, 0
	for i, ch := range s {
		isDigit := '0' <= ch && ch <= '9' // ch 是 0-9 的数字
		if isDigit {
			num = num*10 + int(ch-'0') //字符转数字
		}
		if !isDigit && ch != ' ' || i == len(s)-1 { // ch 是运算符
			switch preSign {
			case '+':
				stack = append(stack, num)
			case '-':
				stack = append(stack, -num)
			case '*':
				stack[len(stack)-1] *= num
			default:
				stack[len(stack)-1] /= num
			}
			preSign = ch
			num = 0
		}
	}
	for _, v := range stack {
		res += v
	}
	return res
}
```


[224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

```go
func calculate(s string) int {
	stack, res, num, sign := []int{}, 0, 0, 1
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			num = num*10 + int(s[i]-'0')
		} else if s[i] == '+' {
			res += sign * num
			num = 0
			sign = 1
		} else if s[i] == '-' {
			res += sign * num
			num = 0
			sign = -1
		} else if s[i] == '(' {
			stack = append(stack, res) //将前一个结果和符号压入栈中
			stack = append(stack, sign)
			res = 0 //将结果设置为0，只需在括号内计算新结果
			sign = 1
		} else if s[i] == ')' {
			res += sign * num
			num = 0
			res *= stack[len(stack)-1]
			res += stack[len(stack)-2]
			stack = stack[:len(stack)-2]
		}
	}
	if num != 0 {
		res += sign * num
	}
	return res
}
```