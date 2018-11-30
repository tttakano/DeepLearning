import sys
import numpy
from argparse import ArgumentParser
from chainer import Chain, ChainList, Variable, cuda, functions, links, optimizer, optimizers, serializers
import util.generators as gens
from util.functions import trace, fill_batch
from util.vocabulary import Vocabulary
import time


def parse_args():
  def_mode="train"
  def_source="../dataset/nctrain_en.txt"
  def_target="../dataset/nctrain_ja.txt"
  def_gpu_device = 0
  def_vocab = 6000
  def_embed = 300
  def_hidden = 300
  def_epoch = 100 
  def_minibatch = 64
  def_generation_limit = 128

  p = ArgumentParser(
    description='Attentional neural machine trainslation',
    usage=
      '\n  %(prog)s train [options] source target model'
      '\n  %(prog)s test source target model'
      '\n  %(prog)s -h',
  )


  p.add_argument('model', help='[in/out] model file')
  p.add_argument('--mode', default=def_mode, help='\'train\' or \'test\'')
  p.add_argument('--source', default=def_source, help='[in] source corpus')
  p.add_argument('--target', default=def_target, help='[in/out] target corpus')
  p.add_argument('--use-gpu', action='store_true', default=True,
    help='use GPU calculation')
  p.add_argument('--gpu-device', default=def_gpu_device, metavar='INT', type=int,
    help='GPU device ID to be used (default: %(default)d)')
  p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
    help='vocabulary size (default: %(default)d)')
  p.add_argument('--embed', default=def_embed, metavar='INT', type=int,
    help='embedding layer size (default: %(default)d)')
  p.add_argument('--hidden', default=def_hidden, metavar='INT', type=int,
    help='hidden layer size (default: %(default)d)')
  p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int,
    help='number of training epoch (default: %(default)d)')
  p.add_argument('--minibatch', default=def_minibatch, metavar='INT', type=int,
    help='minibatch size (default: %(default)d)')
  p.add_argument('--generation-limit', default=def_generation_limit, metavar='INT', type=int,
    help='maximum number of words to be generated for test input (default: %(default)d)')

  args = p.parse_args()
  return args

class XP:
  __lib = None

  @staticmethod
  def set_library(args):
    if args.use_gpu:
      XP.__lib = cuda.cupy
      cuda.get_device(args.gpu_device).use()
    else:
      XP.__lib = numpy

  @staticmethod
  def __zeros(shape, dtype):
    return Variable(XP.__lib.zeros(shape, dtype=dtype))

  @staticmethod
  def fzeros(shape):
    return XP.__zeros(shape, XP.__lib.float32)

  #add
  @staticmethod 
  def __nonzeros(shape, dtype, val):
    return Variable(val * XP.__lib.ones(shape, dtype=dtype))

  #add
  @staticmethod
  def fnonzeros(shape, val=1):
    return XP.__nonzeros(shape, XP.__lib.float32, val)

  @staticmethod
  def __array(array, dtype):
    return Variable(XP.__lib.array(array, dtype=dtype))

  @staticmethod
  def iarray(array):
    return XP.__array(array, XP.__lib.int32)

  @staticmethod
  def farray(array):
    return XP.__array(array, XP.__lib.float32)

class SrcEmbed(Chain):  #add
  def __init__(self, vocab_size, embed_size):
    super(SrcEmbed, self).__init__(
        xe = links.EmbedID(vocab_size, embed_size),
    )

  def __call__(self, x):
    return functions.tanh(self.xe(x))

class Encoder(Chain):
  def __init__(self, embed_size, hidden_size):
    super(Encoder, self).__init__(
        xh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
    )

  def __call__(self, x, c, h):
    return functions.lstm(c, self.xh(x) + self.hh(h))

class Attention(Chain):  #add
  def __init__(self, hidden_size):
    super(Attention, self).__init__(
        aw = links.Linear(hidden_size, hidden_size),
        bw = links.Linear(hidden_size, hidden_size),
        pw = links.Linear(hidden_size, hidden_size),
        we = links.Linear(hidden_size, 1),
    )
    self.hidden_size = hidden_size

  def __call__(self, a_list, b_list, p):
    batch_size = p.data.shape[0]
    e_list = []
    sum_e = XP.fzeros((batch_size, 1))
    for a, b in zip(a_list, b_list):
      w = functions.tanh(self.aw(a) + self.bw(b) + self.pw(p))
      e = functions.exp(self.we(w))
      e_list.append(e)
      sum_e += e
    ZEROS = XP.fzeros((batch_size, self.hidden_size))
    aa = ZEROS
    bb = ZEROS
    for a, b, e in zip(a_list, b_list, e_list):
      e /= sum_e
      aa += functions.reshape(functions.batch_matmul(a, e), (batch_size, self.hidden_size))
      bb += functions.reshape(functions.batch_matmul(b, e), (batch_size, self.hidden_size))
    return aa, bb

class Decoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Decoder, self).__init__(
        ye = links.EmbedID(vocab_size, embed_size),
        eh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
        ah = links.Linear(hidden_size, 4 * hidden_size),
        bh = links.Linear(hidden_size, 4 * hidden_size),
        hf = links.Linear(hidden_size, embed_size),
        fy = links.Linear(embed_size, vocab_size),
    )

  def __call__(self, y, c, h, a, b):
    e = functions.tanh(self.ye(y))
    c, h = functions.lstm(c, self.eh(e) + self.hh(h) + self.ah(a) + self.bh(b))
    f = functions.tanh(self.hf(h))
    return self.fy(f), c, h

class AttentionMT(Chain): #changed
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(AttentionMT, self).__init__(
        emb = SrcEmbed(vocab_size, embed_size),
        fenc = Encoder(embed_size, hidden_size),
        benc = Encoder(embed_size, hidden_size),
        att = Attention(hidden_size),
        dec = Decoder(vocab_size, embed_size, hidden_size),
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

  def reset(self, batch_size):
    self.zerograds()
    self.x_list = []

  def embed(self, x):
    self.x_list.append(self.emb(x))

  def encode(self):
    src_len = len(self.x_list)
    batch_size = self.x_list[0].data.shape[0]
    ZEROS = XP.fzeros((batch_size, self.hidden_size))
    c = ZEROS
    a = ZEROS
    a_list = []
    for x in self.x_list:
      c, a = self.fenc(x, c, a)
      a_list.append(a)
    c = ZEROS
    b = ZEROS
    b_list = []
    for x in reversed(self.x_list):
      c, b = self.benc(x, c, b)
      b_list.insert(0, b)
    self.a_list = a_list
    self.b_list = b_list
    self.c = ZEROS
    self.h = ZEROS

  def decode(self, y):
    aa, bb = self.att(self.a_list, self.b_list, self.h)
    y, self.c, self.h = self.dec(y, self.c, self.h, aa, bb)
    return y

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.vocab_size, file=fp)
      print(self.embed_size, file=fp)
      print(self.hidden_size, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      vocab_size = int(next(fp))
      embed_size = int(next(fp))
      hidden_size = int(next(fp))
      return AttentionMT(vocab_size, embed_size, hidden_size)

def forward(src_batch, trg_batch, src_vocab, trg_vocab, attmt, is_training, generation_limit):
  batch_size = len(src_batch)
  src_len = len(src_batch[0])
  trg_len = len(trg_batch[0]) if trg_batch else 0
  src_stoi = src_vocab.stoi
  trg_stoi = trg_vocab.stoi
  trg_itos = trg_vocab.itos
  attmt.reset(batch_size)

  x = XP.iarray([src_stoi('<s>') for _ in range(batch_size)])
  attmt.embed(x)
  for l in range(src_len):
    x = XP.iarray([src_stoi(src_batch[k][l]) for k in range(batch_size)])
    attmt.embed(x)
  x = XP.iarray([src_stoi('</s>') for _ in range(batch_size)])
  attmt.embed(x)

  attmt.encode()

  t = XP.iarray([trg_stoi('<s>') for _ in range(batch_size)])
  hyp_batch = [[] for _ in range(batch_size)]

  if is_training:
    loss = XP.fzeros(())
    for l in range(trg_len):
      y = attmt.decode(t)
      t = XP.iarray([trg_stoi(trg_batch[k][l]) for k in range(batch_size)])
      loss += functions.softmax_cross_entropy(y, t)
      output = cuda.to_cpu(y.data.argmax(1))
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
    return hyp_batch, loss
  
  else:
    while len(hyp_batch[0]) < generation_limit:
      y = attmt.decode(t)
      output = cuda.to_cpu(y.data.argmax(1))
      t = XP.iarray(output)
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
      if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)):
        break

    return hyp_batch

def train(args):
  trace('making vocabularies ...')
  src_vocab = Vocabulary.new(gens.word_list(args.source), args.vocab)
  trg_vocab = Vocabulary.new(gens.word_list(args.target), args.vocab)

  trace('making model ...')
  attmt = AttentionMT(args.vocab, args.embed, args.hidden)
  if args.use_gpu:
    attmt.to_gpu()

  logs = []  # log
  for epoch in range(args.epoch):
    trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
    trained = 0
    gen1 = gens.word_list(args.source)
    gen2 = gens.word_list(args.target)
    gen3 = gens.batch(gens.sorted_parallel(gen1, gen2, 100 * args.minibatch), args.minibatch)
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(attmt)
    opt.add_hook(optimizer.GradientClipping(5))
    sum_loss = 0  # log
    m = 0  # log
    avr_loss = 0  # log

    for src_batch, trg_batch in gen3:
      if src_batch[0] == []:  # add
        continue
      m += 1  # log
      src_batch = fill_batch(src_batch)
      trg_batch = fill_batch(trg_batch)
      K = len(src_batch)
      n = len(trg_batch[0]) # log
      hyp_batch, loss = forward(src_batch, trg_batch, src_vocab, trg_vocab, attmt, True, 0)
      loss.backward()
      opt.update()
      sum_loss += loss.data / n  # log

      for k in range(K):
        trace('epoch %3d/%3d, sample %8d' % (epoch + 1, args.epoch, trained + k + 1))
        trace('  src = ' + ' '.join([x if x != '</s>' else '*' for x in src_batch[k]]))
        trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in trg_batch[k]]))
        trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[k]]))

      trained += K

    trace('saving model ...')
    prefix = args.model + '.%03.d' % (epoch + 1)
    src_vocab.save(prefix + '.srcvocab')
    trg_vocab.save(prefix + '.trgvocab')
    attmt.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', attmt)
    avr_loss = sum_loss / m  # log
    log = {"epoch": epoch, "loss": avr_loss}  # log
    logs.append(log)  # log

  trace('finished.')

  df = pd.DataFrame(logs) # log
  df.to_csv("%s.csv" % (args.model))  # log


def test(args):
  trace('loading model ...')
  src_vocab = Vocabulary.load(args.model + '.srcvocab')
  trg_vocab = Vocabulary.load(args.model + '.trgvocab')
  attmt = AttentionMT.load_spec(args.model + '.spec')
  if args.use_gpu:
    attmt.to_gpu()
  serializers.load_hdf5(args.model + '.weights', attmt)

  trace('generating translation ...')
  generated = 0

  with open(args.target, 'w') as fp:
    for src_batch in gens.batch(gens.word_list(args.source), args.minibatch):
      if src_batch[0] == []:  # add
        print("", file=fp)
        continue

      src_batch = fill_batch(src_batch)
      K = len(src_batch)

      trace('sample %8d - %8d ...' % (generated + 1, generated + K))
      hyp_batch = forward(src_batch, None, src_vocab, trg_vocab, attmt, False, args.generation_limit)

      for hyp in hyp_batch:
        hyp.append('</s>')
        hyp = hyp[:hyp.index('</s>')]
        print(' '.join(hyp), file=fp)

      generated += K

  trace('finished.')

def main():
  args = parse_args()
  XP.set_library(args)
  start = time.time()
  if args.mode == 'train': train(args)
  elif args.mode == 'test': test(args)
  elapsed_time = time.time() - start
  f = open('take.txt', 'w')
  f.write(str(elapsed_time)) 
  f.close()

if __name__ == '__main__':
  main()
