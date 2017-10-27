package nlp.intent.toolkit;

import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.namefind.*;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.*;
import opennlp.tools.util.featuregen.AdaptiveFeatureGenerator;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

class FileInputStreamFactory implements InputStreamFactory {

    private String pathToFile;

    public FileInputStreamFactory(String path) {
        this.pathToFile = path;
    }

    public InputStream createInputStream() throws IOException {
        return new FileInputStream(this.pathToFile);
    }
}

public class IntentTrainer {

    public static void main(String[] args) throws Exception {

        File trainingDirectory = new File(args[0]);
        String[] slots = new String[0];
        if(args.length > 1){
            slots = args[1].split(",");
        }


        if(!trainingDirectory.isDirectory()) {
            throw new IllegalArgumentException("TrainingDirectory is not a directory: " + trainingDirectory.getAbsolutePath());
        }

        List<ObjectStream<DocumentSample>> categoryStreams = new ArrayList<ObjectStream<DocumentSample>>();
        for (File trainingFile : trainingDirectory.listFiles()) {
            String intent = trainingFile.getName().replaceFirst("[.][^.]+$", "");
            ObjectStream<String> lineStream = new PlainTextByLineStream(new FileInputStreamFactory(trainingFile.getPath()), "UTF-8");
            ObjectStream<DocumentSample> documentSampleStream = new IntentDocumentSampleStream(intent, lineStream);
            categoryStreams.add(documentSampleStream);
        }


        List<DocumentSample> samples = new ArrayList<DocumentSample>();
        DocumentSample sample;

        for (ObjectStream<DocumentSample> item : categoryStreams){
            while ((sample = item.read()) != null){
                samples.add(sample);
            }
        }

        ObjectStream<DocumentSample> combinedDocumentSampleStream = ObjectStreamUtils.createObjectStream(samples);

        TrainingParameters parameters = new TrainingParameters();
        DoccatFactory factory = new DoccatFactory();

        DoccatModel doccatModel = DocumentCategorizerME.train("en", combinedDocumentSampleStream, parameters, factory);
        combinedDocumentSampleStream.close();

        List<TokenNameFinderModel> tokenNameFinderModels = new ArrayList<TokenNameFinderModel>();

        for(String slot : slots) {
            List<ObjectStream<NameSample>> nameStreams = new ArrayList<ObjectStream<NameSample>>();
            for (File trainingFile : trainingDirectory.listFiles()) {
                ObjectStream<String> lineStream = new PlainTextByLineStream(new FileInputStreamFactory(trainingFile.getPath()), "UTF-8");
                ObjectStream<NameSample> nameSampleStream = new NameSampleDataStream(lineStream);
                nameStreams.add(nameSampleStream);
            }

            List<NameSample> nameSamples = new ArrayList<NameSample>();
            NameSample nameSample;

            for (ObjectStream<NameSample> item : nameStreams){
                while ((nameSample = item.read()) != null){
                    nameSamples.add(nameSample);
                }
            }


            ObjectStream<NameSample> combinedNameSampleStream = ObjectStreamUtils.createObjectStream(nameSamples);
            TokenNameFinderFactory nfFactory = new TokenNameFinderFactory();

            TokenNameFinderModel tokenNameFinderModel = NameFinderME.train("en", "city", combinedNameSampleStream, TrainingParameters.defaultParams(), nfFactory);
            combinedNameSampleStream.close();
            tokenNameFinderModels.add(tokenNameFinderModel);
        }


        DocumentCategorizerME categorizer = new DocumentCategorizerME(doccatModel);
        NameFinderME[] nameFinderMEs = new NameFinderME[tokenNameFinderModels.size()];
        for(int i = 0; i < tokenNameFinderModels.size(); i++) {
            nameFinderMEs[i] = new NameFinderME(tokenNameFinderModels.get(i));
        }

        System.out.println("Training complete. Ready.");
        System.out.print(">");

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        while(true){

            String s = reader.readLine();

            String[] source = { s };
            double[] outcome = categorizer.categorize(source);
            Map<String, Double> scores = categorizer.scoreMap(source);

            for (String category : scores.keySet()){
                System.out.println("Category=" + category + " score=" + scores.get(category));
            }

            System.out.print("action=" + categorizer.getBestCategory(outcome) + " args={ ");

            String[] tokens = WhitespaceTokenizer.INSTANCE.tokenize(s);
            for (NameFinderME nameFinderME : nameFinderMEs) {
                Span[] spans = nameFinderME.find(tokens);
                String[] names = Span.spansToStrings(spans, tokens);
                for (int i = 0; i < spans.length; i++) {
                    System.out.print(spans[i].getType() + "=" + names[i] + " ");
                }
            }
            System.out.println("}");
            System.out.print(">");

        }
    }

}
