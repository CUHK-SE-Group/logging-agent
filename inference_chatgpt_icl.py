

import time
import requests
import pandas as pd
from read_data import get_one_prompt
from read_data import get_instrucion
import concurrent.futures

#your openai key
Authorization = ""

# prepare ICL examples:
def get_example(task_name, lineID=None):
    example1 = ""
    example_label1 = ""
    example2 = ""
    example_label2 = ""
    example3 = ""
    example_label3 = ""
    example_prompt = ""
    # predict line ID of logging statement needed
    if task_name == "task1":
        example1 = "public class A { <line0> @Override <line1> public void appendJsonLog(ActionData actionData) { <line2> if (isJsonLogActive()) { <line3> String dir = ""gamelogsJson""; <line4> File saveDir = new File(dir); <line5> if (!saveDir.exists()) { <line6> saveDir.mkdirs(); <line7> } <line8> actionData.sessionId = getSessionId(); <line9> String logFileName = dir + File.separator + ""game-"" + actionData.gameId + "".json""; <line10> try (PrintWriter out = <line11> new PrintWriter(new BufferedWriter(new FileWriter(logFileName, true)))) { <line12> out.println(actionData.toJson()); <line13> } catch (IOException e) { <line14> } <line15> } <line16> } <line17> } <line18> "
        example_label1 = "<line14>"
        example2 = "public class A { <line0> @Override <line1> public void streamClosed(Stream s) { <line2> } <line3> } <line4>"
        example_label2 = "<line2>"
        example3 = "public class A { <line0> @Override <line1> public void merge(Collection<W> toBeMerged, W mergeResult) { <line2> if (LOG.isDebugEnabled()) { <line3> } <line4> mergeResults.put(mergeResult, toBeMerged); <line5> } <line6> } <line7>"
        example_label3 = "<line3>"
    # predict log level
    if task_name == "task2":
        example1 = "public class A { <line0> @Override <line1> public void onError(FacebookException error) { <line2> logger.UNKNOWN(error); <line3> } <line4> } <line5>"
        example_label1 = "error"
        example2 = "public class A { <line0> @Override <line1> public void info(Marker marker, String string, Object o) { <line2> logger.UNKNOWN(marker, string, o); <line3> } <line4> } <line5>"
        example_label2 = "info"
        example3 = "public class A { <line0> public void closed() { <line1> if (LOG.isDebugEnabled()) LOG.UNKNOWN(""NullCallback closed""); <line2> } <line3> } <line4>"
        example_label3 = "debug"
    # predict log message
    if task_name == "task3":
        example1 = "public class A { <line0> @Override <line1> protected void onEntityChange(Entity member) { <line2> log.isDebugEnabled(UNKNOWN); <line3> ((CassandraDatacenterImpl) entity).update(); <line4> } <line5> } <line6>"
        example_label1 = "\"Node {} updated in Cluster {}\", member, this"
        example2 = "public class A { <line0> public static void unregister(DependencyProvider provider) { <line1> log.debug(UNKNOWN); <line2> providers.remove(provider); <line3> } <line4> } <line5>"
        example_label2 = "\"Unregistering \" + provider"
        example3 = "public class A { <line0> @Override <line1> public void onNext(Empty value) { <line2> logger.info(UNKNOWN); <line3> } <line4> } <line5>"
        example_label3 = "\"onNext. message:{}\", value"
    # given pos, generate logging statement (level,msg)
    if task_name == "task4":
        example1 = "public class A { <line0> @Override <line1> public void warn(String msg, Throwable t) { <line2> warnMessages.add(new LogMessage(null, msg, t)); <line3> } <line4> } <line5>"
        example_label1 = "logger.warn(msg, t)"
        example2 = "public class A { <line0> @Override <line1> protected void after() { <line2> if (server != null) { <line3> server.close(CloseMode.GRACEFUL); <line4> } <line5> } <line6> } <line7>"
        example_label2 = "log.debug(\"Shutting down test server\")"
        example3 = "public class A { <line0> private void logError(String errorMessage) { <line1> inputSourceParseErrors.add(errorMessage); <line2> } <line3> } <line4>"
        example_label3 = "LOGGER.error(errorMessage)"
    # generate logging statement (pos,level,msg)
    if task_name == "task5":
        example1 = "public class A { <line0> @AfterClass <line1> public static void afterClass() { <line2> try { <line3> standaloneConsul.stop(); <line4> } catch (Exception e) { <line5> } <line6> } <line7> } <line8>"
        example_label1 = "<line5>      logger.info(""Failed to stop standalone consul"");"
        example2 = "public class A { <line0> @Override <line1> public void onError(FacebookException error) { <line2> } <line3> } <line4>"
        example_label2 = "<line2>    logger.error(error);"
        example3 = "public class A { <line0> public void finish() throws IOException { <line1> this.data.rewind(); <line2> try { <line3> pieceStorage.savePiece(index, this.data.array()); <line4> } finally { <line5> this.data = null; <line6> } <line7> } <line8> } <line9>"
        example_label3 = "<line2>    logger.trace(""Recording {}..."", this);"
    example_prompt = "Example: " + example1 + "\n Label: " + example_label1 + "\n \n" + "Example: " + example2 + "\n Label: " + example_label2 + "\n \n" + "Example: " + example3 + "\n Label: " + example_label3 + "\n \n"
    return example_prompt



def chat_with_gpt(instruction,input):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": Authorization
    }

    data = {
        "model": "gpt-4",
        # change model here
        #"model": "gpt-3.5-turbo", 
        "messages": [
            {
                "role": "system",
                "content": instruction
            },
            {
                "role": "user",
                "content": input
            }
        ]
    }
    # print("data: \n",data)
    response = requests.post(url, json=data, headers=headers)
    output = response.json()

    return output



output_list = []
query_list = []
task_list = []
groundtruth_list = []

data_path = "./task1-5/mixtasks1-4_test.tsv"
# better to prepare a copy version, because the api is unstable and may not run all samples at once.
# data_path = "./task1-5/mixtasks1-4_test_copy.tsv"
result_path = "./task1-5/chatgpt4_result.tsv"
# result_path = "./task1-5/chatgpt3_result.tsv"

df = pd.read_csv(data_path, sep='\t')
list_data_dict = df.to_dict('records')

count = 0
batch_count = 0
batch_size = 5  # Set batch size
max_retries = 3  # Set maximum number of retries

for row in list_data_dict:
    instruction = get_instrucion(row['task'])
    example_prompt = get_example(row['task'])
    input_text = example_prompt + "Query: " + row['code'] + "\n" + "Label:"
    retries = 0
    response = None

    while retries < max_retries:
        response = chat_with_gpt(instruction, input_text)
        if response and 'choices' in response and response['choices'] and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
            break
        retries += 1
        time.sleep(3)
        print(f"Retrying request {retries} for row {count}")
    
    if not response or 'choices' not in response or not response['choices'] or 'message' not in response['choices'][0] or 'content' not in response['choices'][0]['message']:
        print(f"Failed to get valid response for row {count}")
        continue
    
    groundtruth = row['label']
    count += 1
    print(count)
    output = repr(response['choices'][0]['message']['content'])

    output_list.append(output)
    query_list.append(row['code'])
    task_list.append(row['task'])
    groundtruth_list.append(groundtruth)

    batch_count += 1
    if batch_count == batch_size:
        df_result = pd.DataFrame({'task': task_list, 'prompt': query_list, 'label': groundtruth_list, 'predict': output_list})
        df_result.to_csv(result_path, sep='\t', index=False, mode='a', header=False)
        
        # Delete processed batch from original data
        df = df.drop(df.index[:batch_count])
        df.to_csv(data_path, sep='\t', index=False)

        # Reset lists and batch count
        output_list = []
        query_list = []
        task_list = []
        groundtruth_list = []
        batch_count = 0

if batch_count > 0:
    df_result = pd.DataFrame({'task': task_list, 'prompt': query_list, 'label': groundtruth_list, 'predict': output_list})
    df_result.to_csv(result_path, sep='\t', index=False, mode='a', header=False)
    df = df.drop(df.index[:batch_count])
    df.to_csv(data_path, sep='\t', index=False)

print("done")

