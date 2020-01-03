import java.util.Scanner;
public class FizzBuzz1 {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);
        System.out.println("Give me three numbers");
        int num1 = scnr.nextInt();
        int num2 = scnr.nextInt();
        int num3 = scnr.nextInt();
        //If the number is divisible by num1 and num2, the program will output FizzBuzz.
        for (int i = 0; i < num3; i ++) {
            if (((i % num1) == 0) && ((i % num2) == 0)) {
                System.out.println("FizzBuzz");
            }
            //If it is only divisible by num1, "Fizz" will output.
                else if ((i % num1) == 0) {
                System.out.println("Fizz");
            }
                //If it is only divisible by num2, "Buzz" will output.
                else if (((i % num2) == 0)) {
                    System.out.println("Buzz");
            }
                else {
                    //otherwise, the number will print.
                    System.out.println(i);
            }
                }

    }
        }
