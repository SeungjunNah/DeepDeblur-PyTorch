def set_template(args):
    if args.template.find('gopro') >= 0:
        args.dataset = 'GOPRO_Large'
        args.milestones = [500, 750, 900]
        args.end_epoch = 1000
    elif args.template.find('reds') >= 0:
        args.dataset = 'REDS'
        args.milestones = [100, 150, 180]
        args.end_epoch = 200
