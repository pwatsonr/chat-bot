import com.huggingface.transformers.GPT2LMHeadModel;
import com.huggingface.transformers.Tokenizer;
import com.huggingface.transformers.TransformersTokens;
import java.io.IOException;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ChatBotController {

  private final Tokenizer tokenizer;
  private final GPT2LMHeadModel model;

  public ChatBotController() throws IOException {
    this.tokenizer = TransformersTokens.getTokenizer("facebook/opt-250m-finetuned-openai-detectron2");
    this.model = TransformersTokens.getModel("facebook/opt-250m-finetuned-openai-detectron2");
  }

  @GetMapping("/")
  public String prePrompt() {
    return "Welcome to the chatbot. How can I help you today?";
  }

  @PostMapping("/")
  public String generateResponse(@RequestBody String input) {
    String[] inputTokens = tokenizer.tokenize(input).getTokens();
    String[] response = model.generate(inputTokens);
    return tokenizer.decode(response);
  }
}
