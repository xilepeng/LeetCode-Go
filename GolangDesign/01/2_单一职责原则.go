package main

import "fmt"

type ClothersShop struct{}

type ClothersWork struct{}

func (cs *ClothersShop) Style() {
	fmt.Println("\t逛街的装扮...")
}

func (cw *ClothersWork) Style() {
	fmt.Println("\t工作的装扮...")
}

func main() {
	fmt.Println("遵守单一职责原则:")

	fmt.Println("1.逛街业务:")
	cs := ClothersShop{}
	cs.Style()

	fmt.Println("2.工作业务:")
	cw := ClothersWork{}
	cw.Style()

	fmt.Println("3.逛街想穿工作装扮业务")
}
