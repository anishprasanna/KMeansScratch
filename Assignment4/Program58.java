import java.util.Scanner;
public class CountInputLength {
    public static void main (String[]args){
        System.out.println("Count Input Length\n");

        //Variables and Scanner
        Scanner sc = new Scanner (System.in);
        System.out.print("Enter a line of text: ");
        String userInput = sc.nextLine();

        int size = userInput.length();
        int i = 0;

        //For loop. Will subtract 1 from length every time char equals to ' ', '.' or ','
        for (i = 0; i < userInput.length(); i++){
            if ((userInput.charAt(i) == ' ') || (userInput.charAt(i) == '.' )|| (userInput.charAt(i) == ',')){
                size = size - 1;
            }
        }
         System.out.println("The text length without spaces, periods, or commas is: " + size);

    }
}
