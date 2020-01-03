import java.util.Scanner;
public class FizzBuzzGame {
    public static void main(String[] args) {
        Scanner scnr = new Scanner(System.in);

        System.out.println("Fizz Buzz Game ==============\n");
        System.out.print("Input three integers: ");
        int num1 = scnr.nextInt();
        int num2 = scnr.nextInt();
        int num3 = scnr.nextInt();

        for(int i =1; i < num3+1; i++){
            int n = i;
            if(n % num1 == 0 && n % num2 ==0){
                System.out.println("FizzBuzz");
            }
            else if (n% num1 ==0){
                System.out.println("Fizz");
            }
            else if(n % num2 == 0){
                System.out.println("Buzz");
            }
            else{
                System.out.println(n);
            }
        }

    }
}
