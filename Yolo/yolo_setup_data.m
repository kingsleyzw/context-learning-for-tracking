function imdb = yolo_setup_data(varargin)
% Setup PASCAL VOC 2007 data for training
%
opts.dataDir = 'data' ;
opts.addFlipped = true ;
opts.useDifficult = true ;
opts = vl_argparse(opts, varargin) ;
addpath(fullfile(opts.dataDir, 'VOCdevkit', 'VOCcode'));

% Initialize VOC options
VOCinit ;
VOCopts.dataset = 'VOC2007';
opts.dataDir(opts.dataDir=='\')='/';
VOCopts.imgsetpath = [opts.dataDir '/VOCdevkit/VOC2007/ImageSets/Main/%s.txt'];
VOCopts.annopath  = [opts.dataDir '/VOCdevkit/VOC2007/Annotations/%s.xml'];

imdb.classes.name = VOCopts.classes ;
imdb.classes.description = VOCopts.classes ;
imdb.imageDir = fullfile(opts.dataDir, 'VOCdevkit', 'VOC2007' ,'JPEGImages') ;
% -------------------------------------------------------------------------
% read Images
% -------------------------------------------------------------------------
k = 0 ;
for thisSet = {'train', 'val', 'test'}
  thisSet = char(thisSet) ;

  fprintf('Loading PASCAL VOC %s set\n', thisSet) ;
  VOCopts.testset = thisSet ;

  [gtids,t]=textread(sprintf(VOCopts.imgsetpath,thisSet),'%s %d');

  k = k + 1 ;
  imdb_.images.name{k} = strcat(gtids,'.jpg');
  imdb_.images.set{k}  = k * ones(size(imdb_.images.name{k}));
  imdb_.images.size{k} = zeros(numel(imdb_.images.name{k}),2);
  imdb_.boxes.gtbox{k} = cell(size(imdb_.images.name{k}));
  imdb_.boxes.gtlabel{k} = cell(size(imdb_.images.name{k}));
  imdb_.boxes.truth{k} = cell(size(imdb_.images.name{k}));
  % Load ground truth objects
  for i=1:length(gtids)
    % Read annotation.
    rec=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    imdb_.images.size{k}(i,:) = rec.imgsize(1:2);
    % extract objects of class
    BB = vertcat(rec.objects(:).bbox);
    [~,label]=ismember({rec.objects(:).class},VOCopts.classes);
    if ~isempty(BB)
      imdb_.boxes.gtbox{k}{i} = BB;
      assert(all(BB(:,3)<=rec.imgsize(1)));
      assert(all(BB(:,4)<=rec.imgsize(2)));
      imdb_.boxes.gtlabel{k}{i} = label;
      imdb_.boxes.truth{k}{i} = data_generate(imdb_.boxes.gtbox{k}{i},imdb_.boxes.gtlabel{k}{i},imdb_.images.size{k}(i,:));
    end
  end
end
imdb.images.name = vertcat(imdb_.images.name{:}) ;
imdb.images.size = vertcat(imdb_.images.size{:}) ;
imdb.images.set  = vertcat(imdb_.images.set{:}) ;
imdb.boxes.gtbox = vertcat(imdb_.boxes.gtbox{:}) ;
imdb.boxes.gtlabel = vertcat(imdb_.boxes.gtlabel{:}) ;
imdb.boxes.truth = vertcat(imdb_.boxes.truth{:}) ;
% -------------------------------------------------------------------------
%  Postprocessing
% -------------------------------------------------------------------------
[~,si] = sort(imdb.images.name) ;
imdb.images.name = imdb.images.name(si) ;
imdb.images.set = imdb.images.set(si) ;
imdb.images.size = imdb.images.size(si,:) ;
imdb.boxes.gtbox = imdb.boxes.gtbox(si)' ;
imdb.boxes.gtlabel = imdb.boxes.gtlabel(si) ;
imdb.boxes.truth = imdb.boxes.truth(si) ;



