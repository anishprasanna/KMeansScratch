import java.util.Scanner;
public class FizzBuzzGame {
    public static void main (String[] arg){
        System.out.println("Fizz Buzz Game =============");
        Scanner key = new Scanner(System.in);
        System.out.println("Input three integers(X,Y,Z): ");
        int X = key.nextInt();
        int Y = key.nextInt();
        int N = key.nextInt();
        for(int i = 1; i <= N; i++){
            if((i%X == 0) && (i%Y == 0)){
                System.out.println("FizzBuzz");
            }else if(i%X == 0){
                System.out.println("Fizz");
            }else if(i%Y==0){
                System.out.println("Buzz");
            }else System.out.println(i);

        }



    }
}
