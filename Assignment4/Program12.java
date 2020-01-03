import java.util.Random;
public class SumRandomIntegers {
    public static void main(String []arg){
        Random rnum = new Random();


        int number;
        int i;
        int sum=0;
        double average;
        int max=-1;

        for(i= 0; i<100; i++){
            number = rnum.nextInt(1000);
            sum = sum + number;
            if (number > max){
                max =number;
            }
        }
        System.out.println("The sum of random 100 integers was "+ sum);
        average = sum/100.00;
        System.out.printf("The average of 100 random integers was %.2f\n", average);
        System.out.println("The largest number among 100 random integers was " + max);
    }
}
