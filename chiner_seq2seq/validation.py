#!/usr/bin/env python

import argparse

from nltk.translate import bleu_score
import numpy
import progressbar
import six

import chainer
from chainer import serializers
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import matplotlib as mpl
mpl.use('Agg')


UNK = 0
EOS = 1


'''
ex)
xs=[[i, am, taro],[i, was, born, in, japan],[i, love, music],[how,are,you]]
ex=[i, am, taro,i, was, born, in, japan,i, love, music,how,are,you]
exs=[[i, am, taro],[i, was, born, in, japan],[i, love, music],[how,are,you]](embed by neural network)
'''
def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        xs = [x[::-1] for x in xs]                                  #reverse input      ["i", "am", "taro"] →["taro", "am", "I"]

        eos = self.xp.array([EOS], numpy.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]      #[eos,y1,y2,...]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]     #[y1,y2,...,eos]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, _ = self.encoder(None, None, exs)           #hidden_state, p_param, encoder(hiddenn_state, preb_c, words)
        _, _, os = self.decoder(hx, cx, eys)                     #####todo?

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=50):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)                                            #vector to id
                ys = self.xp.argmax(wy.data, axis=1).astype(numpy.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):             #todo change batch size
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--SOURCE', default="./dataset/train.en", help='source sentence list')
    parser.add_argument('--TARGET', default="./dataset/train.jp", help='target sentence list')
    parser.add_argument('--SOURCE_VOCAB', default="./dataset/vocab.en", help='source vocabulary file')
    parser.add_argument('--TARGET_VOCAB', default="./dataset/vocab.jp", help='target vocabulary file')
    parser.add_argument('--validation-source', default="./dataset/test.en",
                        help='source sentence list for validation')
    parser.add_argument('--validation-target', default="./dataset/test.jp",
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--start', '-s', default=93,
                        help='start test from this column')
    parser.add_argument('--model', '-m', default='./result/300-200/snapshot_epoch-200',
                        help='model name to validate')

    args = parser.parse_args()

    source_ids = load_vocabulary(args.SOURCE_VOCAB)      #dict      {a:11, the:3, ...}      word_to_id
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    target_words = {i: w for w, i in target_ids.items()}            #dict      {11:a, 3:the, ...}      id_to_wod
    source_words = {i: w for w, i in source_ids.items()}

    model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)
    serializers.load_npz('./result/300-200/snapshot_epoch-200', model, path='updater/model:main/')

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    if args.validation_source and args.validation_target:
        test_source = load_data(source_ids, args.validation_source)
        test_target = load_data(target_ids, args.validation_target)
        assert len(test_source) == len(test_target)
        test_data = list(six.moves.zip(test_source, test_target))
        test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
        test_source_unknown = calculate_unknown_ratio(
            [s for s, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        print('Validation source unknown ratio: %.2f%%' %
              (test_source_unknown * 100))
        print('Validation target unknown ratio: %.2f%%' %
              (test_target_unknown * 100))

        with open("./translate.txt", "w") as f:
            f.write('')
            
        for i in range(100):
            if i % 10 == 0:
                print(i)
            start = args.start + i * args.batchsize
            test = test_data[start : start + args.batchsize]
            source = [cuda.cupy.asarray(s) for s, _ in test]  # [[sentence1], [sentence2], [sentence3], ...]
            target = [cuda.cupy.asarray(t) for _, t in test]
            result = model.translate(source)
            with open("./translate.txt", "a") as f:
                for b in range(args.batchsize):
                    source_sentence = ' '.join([source_words[int(x)] for x in source[b]])
                    target_sentence = ' '.join([target_words[int(y)] for y in target[b]])
                    result_sentence = ' '.join([target_words[int(y)] for y in result[b]])
                    f.write('# source[{0}] : {1}{2}'.format(b, source_sentence, '\n'))
                    f.write('# result[{0}] : {1}{2}'.format(b, result_sentence, '\n'))
                    f.write('# expect[{0}] : {1}{2}'.format(b, target_sentence, '\n'))
                    f.write('\n')
                f.write('\n')

if __name__ == '__main__':
    main()
