import java.util.Scanner;
public class CountInputLength {
    public static void main (String [] arg){
        Scanner key = new Scanner(System.in);
        System.out.println("Count Input Length =============\n");
        System.out.println("Enter a line of text: ");
        String imputLength = key.nextLine();
        int length = imputLength.length();
        int count = 0;
        int i;
        for (i = 0 ; i<length; i++ ){
            if((imputLength.charAt(i) != ',') && (imputLength.charAt(i) != ' ') && (imputLength.charAt(i) != '.')){
                count++;
            }
        }System.out.println("The text length without spaces, periods, or commas is: " + count);

    }
}
