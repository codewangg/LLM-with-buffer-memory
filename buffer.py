from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
from datasets import Dataset


class Buffer:
    """
    A class representing a buffer that stores and manages text samples with their embeddings.
    """
    def __init__(self, encoder='dpr'):
        """
        Initializes a new Buffer instance.

        Args:
            encoder (str): The encoder type to use, e.g., 'dpr'.
        """
        if encoder == 'dpr':
            self.encoder = 'dpr'
            torch.set_grad_enabled(False)
            self.ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        init_sentence = 'hello world'
        init_embedding = self.ctx_encoder(**self.ctx_tokenizer(init_sentence, return_tensors="pt"))[0][0].numpy()
        self.samples = {}
        temp = {'embedding': init_embedding, 'content':init_sentence}
        self.ds = Dataset.from_list([temp])

    def embed(self, input):
        """
        Generates an embedding for the given input text.

        Args:
            input (str): The input text to be embedded.

        Returns:
            numpy.ndarray: The embedding vector of the input text.
        """
        if self.encoder == 'dpr':
            return self.ctx_encoder(**self.ctx_tokenizer(input, return_tensors="pt"))[0][0].numpy()

    def add_sample(self, input):
        """
        Adds a new sample to the buffer along with its embedding.

        Args:
            input (str): The input text of the sample to be added.
        """
        embedding = self.embed(input)
        self.ds = self.ds.add_item({'embedding':embedding, 'content':input})


    def get_nearest_samples(self, query, k=5):
        """
        Retrieves the nearest samples to a given query text based on their embeddings.

        Args:
            query (str): The query text to find nearest samples for.
            k (int): The number of nearest samples to retrieve.

        Returns:
            list: A list of nearest sample contents.
        """
        query_embedding = self.embed(query)
        if not self.ds.is_index_initialized('embedding'):
            self.ds.add_faiss_index(column='embedding')
        scores, samples = self.ds.get_nearest_examples('embedding', query_embedding, k)
        return samples['content']


    def __len__(self):
        """
        Returns the number of samples in the buffer.

        Returns:
            int: The number of samples in the buffer.
        """
        return self.ds.num_rows
    
    def get_samples(self):
        """
        Returns the entire buffer's contents.

        Returns:
            Dataset: The dataset containing the buffer's samples and embeddings.
        """
        return self.ds
    


# Example usage
if __name__ == "__main__":
    buffer = Buffer()

    # Add sample to the buffer
    buffer.add_sample('how are you')
    buffer.add_sample('the weather today is very nice, do you like do some outdoor sports?')
    buffer.add_sample('My favourite sport is snoboarding')
    buffer.add_sample('I like sports very much, I like hiking')
    buffer.add_sample('I had shrimp pizza as dinner')

    print("Buffer contents:")
    for i in range(len(buffer)):
        print(buffer.get_samples()[i]['content'])

    print(buffer.get_nearest_samples('I like outdoor and sports', 2))



