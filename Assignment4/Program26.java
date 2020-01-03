import java.util.Scanner;
public class FizzBuzzGame {
    public static void main (String[] arg){
        //create variables and scanner
        int intX,intY,intN;
        Scanner scnr = new Scanner(System.in);
        //ask for inputs
        System.out.println("Enter three integers: ");
        intX = scnr.nextInt();
        intY = scnr.nextInt();
        intN = scnr.nextInt();
        //calculate which numbers are divisible and print accordingly
        if ((intX >= 1)&&(intX <= 100)&&(intY >= 1)&&(intY <= 100)&&(intN >= 1)&&(intN <= 100)){
            if ((intX < intY)&&(intY <= intN)){
                for(int i = 1; i <= intN; i++){
                    if ((i % intX == 0) &&(i % intY == 0)){
                        System.out.println("FizzBuzz");
                    }else if(i % intX == 0){
                        System.out.println("Fizz");
                    }else if(i % intY == 0){
                        System.out.println("Buzz");
                    }else{
                        System.out.println(i);
                    }
                }
            }
        }
    }
}
