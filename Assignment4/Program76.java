import java.util.Random;
public class SumRandomIntegers {
    public static void main (String[] arg){
        System.out.println("Random Integers =============\n");
        Random key = new Random();
        int sum = 0;
        for (int i = 0; i < 100; i++){
            int num1= key.nextInt(1000);
            sum+= num1;
        }System.out.println("The sum of random 100 integers was " + sum);
        double average = (double) sum/100;
        int largNum = -1;
        System.out.printf("The average of 100 random integers was %.2f\n", average);
       for (int j = 0; j < 100; j++){
           int num2 = key.nextInt(100);
           if(largNum < num2){
               largNum=num2;
           }

       }System.out.println("The largest number among 100 random integers was " + largNum);
    }
}
