package main

import "fmt"

func lengthOfLongestSubstring(s string) int {
	longest, n := 0, len(s)
	freq := make(map[byte]int, n) // 哈希集合记录每个字符出现次数
	for i, j := 0, 0; j < n; j++ {
		freq[s[j]]++         // 首次出现存入哈希
		for freq[s[j]] > 1 { // 循环检测：如果当前字符与首字符重复
			freq[s[i]]--         // 收缩窗口，跳过重复首字符
			i++                  // 向后扫描
			if freq[s[j]] == 1 { // 优化：如果无重复退出循环
				fmt.Println("优化")
				break
			}
		}
		if longest < j-i+1 { // 统计无重复字符的最长子串
			longest = j - i + 1
		}
	}
	return longest
}

func main() {
	s := "aaaaa"
	longest := lengthOfLongestSubstring(s)
	fmt.Println("\"aaaaa\"中无重复字符的最长子串为：", longest)
}
