package main

import "fmt"

func main() {
	/*
		数据存储是以“字节”（Byte）为单位，
		数据传输大多是以“位”（bit，又名“比特”）为单位，一个位就代表一个0或1（即二进制），
		每8个位（bit，简写为b）组成一个字节（Byte，简写为B），是最小一级的信息单位 [4]  。
	*/
	s := "0"
	fmt.Println("字符串的长度（字节个数):", len(s))
	fmt.Println("\"0\"的字节数(byte):", s[0]) // 88 byte 字节

	s1 := '0'
	s2 := 48
	fmt.Printf("获取X的字节数：%d %d %d\n", s[0], s1, s2)

	for i := 0; i < len(s); i++ {
		// fmt.Printf("%d ", s[i])
		fmt.Printf("%c\t", s[i])
	}
	fmt.Printf("\n")

	//字符串是字节的集合
	slice3 := []byte{65, 66, 67}
	s3 := string(slice3) //字节转字符串
	fmt.Println(s3)

	s4 := "abcde"
	slice4 := []byte(s4) //根据字符串获取对应的字节
	fmt.Println(slice4)
}
