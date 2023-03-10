package main

import "fmt"

func main() {
	s := []int(nil)
	s1 := []int{}
	fmt.Printf("%v\n", s)
	fmt.Printf("%v\n", len(s))

	fmt.Printf("%v\n", s1)
	fmt.Printf("%v\n", len(s1))
}
