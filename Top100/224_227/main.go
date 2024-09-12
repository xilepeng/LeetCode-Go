package main

func main() {
	s := "3-(2+1)"
	calculate(s)
}

func calculate(s string) (res int) {
	stack := []int{1} // 栈顶元素记录当前位置所处的每个括号所「共同形成」的符号
	sign := 1         // 标记「当前」的符号，取值为 {−1,+1} 的整数
	n := len(s)
	for i := 0; i < n; {
		switch s[i] {
		case ' ':
			i++
		case '+':
			sign = stack[len(stack)-1]
			i++
		case '-':
			sign = -stack[len(stack)-1]
			i++
		case '(':
			stack = append(stack, sign) // sign 入栈
			i++
		case ')':
			stack = stack[:len(stack)-1] // 出栈
			i++
		default:
			num := 0
			for ; i < n && '0' <= s[i] && s[i] <= '9'; i++ {
				num = num*10 + int(s[i]-'0')
			}
			res += sign * num
		}
	}
	return
}
