
## [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode.cn/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/description/)

```go
func findNthDigit(n int) int {
	start, digit := 1, 1
	for n > 9*start*digit {
		n -= 9 * start * digit
		start *= 10
		digit++
	}
	num := start + (n-1)/digit
	digitIndex := (n - 1) % digit
	return int(strconv.Itoa(num)[digitIndex] - '0')
}

func findNthDigit1(n int) int {
	digit, start, count := 1, 1, 9
	for n > count {
		n -= count
		start *= 10
		digit++
		count = 9 * start * digit
	}
	num := start + (n-1)/digit
	index := (n - 1) % digit
	return int((strconv.Itoa(num)[index]) - '0')
}

func findNthDigit2(n int) int {
	if n <= 9 {
		return n
	}
	bits := 1
	for n > 9*int(math.Pow10(bits-1))*bits {
		n -= 9 * int(math.Pow10(bits-1)) * bits
		bits++
	}
	index := n - 1
	start := int(math.Pow10(bits - 1))
	num := start + index/bits
	digitIndex := index % bits
	return num / int(math.Pow10(bits-digitIndex-1)) % 10
}
```








