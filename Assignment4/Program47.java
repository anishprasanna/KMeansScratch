import java.util.Scanner;
public class FizzBuzzGame {
    public static void main (String [] args){
        Scanner sc = new Scanner(System.in);

        //declare variables to be used in this program
        int num1;
        int num2;
        int bound;

        //prompt the user to enter three integers
        System.out.print( "Input three integers: " );
        num1 = sc.nextInt();
        num2 = sc.nextInt();
        bound = sc.nextInt();

        //run the fizzbuzz game
        for (int i = 1; i <= bound; i++){
            if ((i % num1 == 0) && (i % num2 == 0)){
                System.out.println( "FizzBuzz" );
            }
            else if (i % num1 == 0){
                System.out.println( "Fizz" );
            }
            else if (i % num2 == 0){
                System.out.println( "Buzz" );
            }
            else {
                System.out.println(i);
            }
        }
    }
}
