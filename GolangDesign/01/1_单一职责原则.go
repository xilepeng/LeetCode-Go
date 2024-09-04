package main

import "fmt"

type Clothers struct{}

func (c *Clothers) OnWork() {
	fmt.Println("\t工作的装扮...")
}

func (c *Clothers) OnShop() {
	fmt.Println("\t逛街的装扮...")
}

func main() {
	c := Clothers{}
	fmt.Println("1.未遵守单一职责原则:")
	c.OnWork()
	c.OnShop()

	fmt.Println("2.逛街业务，想穿工作服:")
	c.OnWork()

}
