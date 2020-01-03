import java.util.Scanner;
public class FizzBuzzGame {
    public static void main (String[]args){
        System.out.println("Fizz Buzz Game\n");

        //Variables and scanner
        Scanner sc = new Scanner (System.in);
        System.out.print("Input three integers: ");
        int divisor1 = sc.nextInt();
        int divisor2 = sc.nextInt();
        int range = sc.nextInt();
        int i = 1;

        //Check if number is divisible by both divisors, only one or none to decide what to output
        while(i <= range){
            if (i%(divisor1*divisor2) == 0){
                System.out.println("FizzBuzz");
            }
            else if (i%divisor1 == 0){
                System.out.println("Fizz");
            }
            else if (i%divisor2 == 0){
                System.out.println("Buzz");
            }
            else {
                System.out.println(i);
            }
            i++;
        }
    }
}
